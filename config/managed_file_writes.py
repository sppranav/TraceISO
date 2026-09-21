"""Cross-process write transaction for lab-managed configuration files.

TraceISO and its standalone Qt tools are separate applications that can have
the same managed JSON file open at once. A Python lock would only serialize
threads inside one interpreter, so revision checking and replacement are
serialized here with a real OS file lock: ``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows. Both are released by the kernel when the
holding process exits, so a crash cannot leave the file permanently locked.

The expected revision is re-read *inside* the lock. Checking it before taking
the lock is what let two saves that both observed the same revision each
decide they were current and then overwrite one another.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type

_log = logging.getLogger(__name__)

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


LOCK_TIMEOUT_SECONDS = 20.0
_LOCK_POLL_SECONDS = 0.01


class ManagedWriteLockError(RuntimeError):
    """The managed file stayed locked by another process past the timeout."""


@dataclass(frozen=True)
class ManagedWriteWarning:
    """Housekeeping that failed on a transaction whose primary write committed.

    A backup that could not be maintained is not a failed save, and must not be
    reported as one — the edit is on disk. It is not a clean save either, so it
    cannot be silent, which is what it was: the rotation failure was caught,
    the staged copy unlinked, and the caller told "Saved" with no channel
    through which it could learn otherwise (audit A084).

    ``existing_backup_retained`` is ``True`` only where an earlier backup was
    actually there to keep. On a first save there is nothing to retain and a
    message promising a recovery copy would be wrong exactly when it matters.
    """

    operation: str  # "stage_backup" | "rotate_backup" | "cleanup"
    path: str
    cause: str
    message: str
    existing_backup_retained: Optional[bool] = None


def _record(
    warnings: Optional[List[ManagedWriteWarning]],
    warning: ManagedWriteWarning,
) -> None:
    """Hand a warning to the caller's collector, or log it if there is none."""
    if warnings is None:
        _log.warning("%s: %s", warning.operation, warning.message)
        return
    warnings.append(warning)


def _backup_warning(
    operation: str, backup_path: Path, cause: BaseException, *, had_backup: bool,
) -> ManagedWriteWarning:
    if had_backup:
        tail = (
            "The existing backup was kept. It is a real earlier revision of "
            "this file, but not necessarily the immediately previous one."
        )
    else:
        tail = "There is no recovery copy for this file."
    return ManagedWriteWarning(
        operation=operation,
        path=str(backup_path),
        cause=str(cause),
        message=(
            f"The recovery backup at {backup_path} could not be updated "
            f"({cause}). {tail}"
        ),
        existing_backup_retained=had_backup,
    )


def _best_effort_unlink(
    target: Path, warnings: Optional[List[ManagedWriteWarning]],
) -> None:
    """Remove a staging file without ever masking the caller's real error."""
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        _record(warnings, ManagedWriteWarning(
            operation="cleanup",
            path=str(target),
            cause=str(exc),
            message=(
                f"The staging file {target} could not be removed ({exc}). It "
                "holds bytes from this transaction and can be deleted by hand."
            ),
        ))


def file_content_revision(path: Path) -> str:
    """Return a content revision token for *path*.

    A SHA-256 digest of the bytes on disk. Preferred over the modification
    time because filesystem timestamp resolution can be coarse — on Windows
    the file time comes from a system clock that advances in ~15 ms steps, so
    two saves inside one tick can share an mtime. A missing file has the
    revision ``""``, which is distinct from any real content.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise ValueError(f"Could not read {path} to compute its revision: {exc}.") from exc
    return hashlib.sha256(data).hexdigest()


def _lock_region(handle, *, acquire: bool) -> None:
    if fcntl is not None:
        operation = fcntl.LOCK_EX | fcntl.LOCK_NB if acquire else fcntl.LOCK_UN
        fcntl.flock(handle.fileno(), operation)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows only
        handle.seek(0)
        mode = msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK
        msvcrt.locking(handle.fileno(), mode, 1)
        return
    raise ManagedWriteLockError(  # pragma: no cover - no known platform
        "Neither fcntl nor msvcrt is available; managed writes cannot be serialized."
    )


@contextmanager
def exclusive_file_lock(path: Path, *, timeout: float = LOCK_TIMEOUT_SECONDS):
    """Hold an exclusive cross-process lock covering *path*.

    The lock is taken on a sibling ``<name>.lock`` file rather than on the
    managed file itself, so it survives the atomic replace that swaps the
    managed file's inode.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    handle = open(lock_path, "a+b")
    try:
        while True:
            try:
                _lock_region(handle, acquire=True)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ManagedWriteLockError(
                        f"Timed out after {timeout:g}s waiting for another process to "
                        f"finish writing {path}."
                    )
                time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            try:
                _lock_region(handle, acquire=False)
            except OSError:  # pragma: no cover - best effort release
                pass
    finally:
        handle.close()


def _bump_mtime_if_unchanged(path: Path, previous_mtime_ns: Optional[int]) -> None:
    """Guarantee the modification time actually moved.

    Callers that still pass ``expected_mtime`` rely on the timestamp changing
    when the contents change. Where the clock has not advanced between the
    observation and the replace, force it forward so a competing saver holding
    the old value sees a mismatch rather than deciding it is still current.
    The increment escalates because filesystems quantize timestamps — NTFS to
    100 ns, and older filesystems far more coarsely — so a one-nanosecond step
    can round straight back to the value it was meant to leave behind.
    """
    if previous_mtime_ns is None:
        return
    stat_result = path.stat()
    if stat_result.st_mtime_ns > previous_mtime_ns:
        return
    for increment_ns in (1_000, 1_000_000, 1_000_000_000, 2_000_000_000):
        try:
            os.utime(path, ns=(stat_result.st_atime_ns, previous_mtime_ns + increment_ns))
        except OSError:  # pragma: no cover - revision token still guards the save
            return
        if path.stat().st_mtime_ns > previous_mtime_ns:
            return


def commit_managed_text(
    path: Path,
    text: str,
    *,
    expected_mtime: Optional[float] = None,
    expected_revision: Optional[str] = None,
    conflict_error: Type[Exception],
    conflict_message: str,
    timeout: float = LOCK_TIMEOUT_SECONDS,
    backup_path: Optional[Path] = None,
    warnings: Optional[List[ManagedWriteWarning]] = None,
) -> str:
    """Replace *path* with *text* as one serialized transaction.

    Returns the content revision of what was written, so a caller can hold a
    current token without re-reading the file. Raises *conflict_error* when a
    supplied expectation no longer matches what is on disk.

    When *backup_path* is given, the outgoing revision is rotated into it —
    but only **after** the new primary has committed. Copying the primary onto
    the backup first, as the CRM manager used to, means a failed replacement
    leaves the backup equal to a file that never changed, and the last
    genuinely recoverable revision is gone with no message (audit A084). A
    save that writes identical bytes does not rotate at all, so a run of
    no-change saves cannot walk the recovery copy forward either.

    Backup maintenance is best effort and never fails the save, but it is no
    longer silent: pass *warnings* to collect a :class:`ManagedWriteWarning`
    for each housekeeping step that did not succeed, so a caller can report a
    committed save with a degraded recovery copy as exactly that. Callers that
    pass nothing keep their existing return contract and the warnings are
    logged instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(path, timeout=timeout):
        exists = path.exists()
        previous_mtime_ns: Optional[int] = None
        if exists:
            previous_mtime_ns = path.stat().st_mtime_ns
        if expected_mtime is not None and exists:
            if path.stat().st_mtime != expected_mtime:
                raise conflict_error(conflict_message)
        if expected_revision is not None:
            if file_content_revision(path) != expected_revision:
                raise conflict_error(conflict_message)

        # Written as bytes, not text: text mode would translate newlines on
        # Windows, so the bytes on disk would not match the revision computed
        # here and the two platforms would disagree about a file's revision.
        payload = text.encode("utf-8")

        # Stage the outgoing revision beside the file. This only reads the
        # primary; nothing recoverable is touched until the new primary is in
        # place.
        staged_previous: Optional[Path] = None
        had_backup = backup_path is not None and backup_path.exists()
        if backup_path is not None and exists and path.read_bytes() != payload:
            staged_previous = path.with_name(
                f"{path.stem}_{uuid.uuid4().hex}{path.suffix}.prev"
            )
            try:
                shutil.copy2(path, staged_previous)
            except OSError as exc:
                # No recovery copy is better than destroying the existing one.
                _best_effort_unlink(staged_previous, warnings)
                staged_previous = None
                _record(warnings, _backup_warning(
                    "stage_backup", backup_path, exc, had_backup=had_backup,
                ))

        tmp_path = path.with_name(f"{path.stem}_{uuid.uuid4().hex}{path.suffix}.tmp")
        try:
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
        except Exception:
            # Cleanup is best effort. An unlink that fails here must not
            # replace the write error the caller has to see.
            _best_effort_unlink(tmp_path, warnings)
            if staged_previous is not None:
                _best_effort_unlink(staged_previous, warnings)
            raise

        # The new primary is committed; rotating the recovery copy is now safe.
        if staged_previous is not None:
            try:
                staged_previous.replace(backup_path)
            except OSError as exc:
                # The primary is correct and the previous backup, if there was
                # one, is still a recoverable revision. Leave it alone and say
                # that the rotation did not happen.
                _best_effort_unlink(staged_previous, warnings)
                _record(warnings, _backup_warning(
                    "rotate_backup", backup_path, exc, had_backup=had_backup,
                ))

        _bump_mtime_if_unchanged(path, previous_mtime_ns)

    return hashlib.sha256(payload).hexdigest()
