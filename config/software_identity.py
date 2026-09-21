"""Measured source identity scoped to executable toolkit inputs, without paths."""
import hashlib
import json
from pathlib import Path
import subprocess


def validate_software_identity(value):
    """Validate a recorded identity without substituting this installation."""
    import re
    from pathlib import PurePosixPath
    if not isinstance(value, dict) or value.get('schema') != 'traceiso.software_identity.v1':
        raise ValueError('Unsupported or missing software identity')
    if set(value) != {'schema', 'commit', 'dirty', 'source_sha256', 'files'}:
        raise ValueError('Software identity fields are incomplete or unsupported')
    if not isinstance(value['commit'], str) or (value['commit'] and not re.fullmatch(r'[0-9a-f]{40,64}', value['commit'])):
        raise ValueError('Invalid recorded software commit')
    if value['dirty'] is not None and type(value['dirty']) is not bool:
        raise ValueError('Invalid recorded dirty state')
    if not isinstance(value['files'], dict):
        raise ValueError('Invalid recorded source manifest')
    for name, sha in value['files'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not re.fullmatch(r'[0-9a-f]{64}', str(sha)):
            raise ValueError('Invalid recorded source entry')
    expected = hashlib.sha256(json.dumps(value['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if value['source_sha256'] != expected:
        raise ValueError('Recorded source digest does not match its manifest')
    return value

def software_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = {}
    for folder in ("config", "domain", "file_io", "ui", "tools"):
        for p in sorted((root/folder).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".json", ".qss"} and "__pycache__" not in p.parts:
                files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("TraceISO.py", "__init__.py", "pyproject.toml", "requirements.txt"):
        p=root/name
        if p.is_file(): files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    commit, dirty = "", None
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() == root.resolve():
            commit = git("rev-parse", "HEAD")
            dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.SubprocessError):
        pass
    return {"schema": "traceiso.software_identity.v1", "commit": commit,
            "dirty": dirty, "source_sha256": hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
            "files": files}
