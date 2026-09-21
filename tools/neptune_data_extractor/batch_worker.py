"""QThread workers for non-blocking batch conversion and validation.

These are thin Qt adapters over ``mapper_logic`` — the actual extraction,
HDF5 writing and validation live in (and are tested via) ``mapper_logic`` so
the UI and the test suite exercise the same code path.
"""

from PyQt5.QtCore import QThread, pyqtSignal

from tools.neptune_data_extractor.mapper_logic import (
    BatchOutcome,
    freeze_template_data,
    pre_flight_validation,
    process_batch,
)


class BatchWorker(QThread):
    """Runs batch HDF5 conversion on a background thread.

    ``run_token`` identifies this worker's run so a UI slot can ignore a
    stale/superseded run's completion signal instead of acting on it — the
    caller is expected to pass a fresh, monotonically increasing value per
    run and compare it against the currently active run before applying
    ``finished``/``fatal_error`` results.

    The file list and the mapping are **snapshotted at construction**. The main
    window edits its mapping dictionaries in place, and only Convert/Cancel are
    governed while a batch runs, so a mapping edit made mid-batch would
    otherwise reach the files this worker has not reached yet.
    """

    progress_update = pyqtSignal(int, str, int)  # (percentage, filename, run_token)
    finished = pyqtSignal(BatchOutcome)      # structured conversion outcome
    fatal_error = pyqtSignal(str, int)       # (unrecoverable error, run_token)

    def __init__(self, file_paths, template_data, output_path, run_token=0, parent=None):
        super().__init__(parent)
        self.file_paths = tuple(file_paths)
        self.template_data = freeze_template_data(template_data)
        self.output_path = output_path
        self.run_token = run_token
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            outcome = process_batch(
                self.file_paths,
                self.template_data,
                self.output_path,
                progress_callback=lambda pct, fname: self.progress_update.emit(
                    pct, fname, self.run_token
                ),
                cancel_check=lambda: self._cancelled,
            )
            outcome.run_token = self.run_token
            # Reader verification runs here, never on the GUI thread.
            try:
                from file_io.hdf5_reader import load_hdf5
                result = load_hdf5(outcome.output_path)
                element = getattr(result, "detected_element", None)
                outcome.reader_verification = {
                    "ok": True, "element": getattr(element, "symbol", "unknown"),
                    "samples": len(result.samples), "warnings": list(result.warnings or []),
                }
            except Exception as exc:
                outcome.reader_verification = {"ok": False, "error": str(exc)}
            self.finished.emit(outcome)
        except Exception as e:  # noqa: BLE001 - surface output-file failures to the UI
            self.fatal_error.emit(str(e), self.run_token)


class ValidationWorker(QThread):
    """Runs pre-flight validation on a background thread.

    Snapshots its inputs for the same reason as :class:`BatchWorker`: the
    mapping a batch was cleared to run under must be the mapping that was
    validated.
    """

    progress_update = pyqtSignal(int, str)
    finished = pyqtSignal(list)

    def __init__(self, file_paths, template_data, parent=None):
        super().__init__(parent)
        self.file_paths = tuple(file_paths)
        self.template_data = freeze_template_data(template_data)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            errors = pre_flight_validation(
                self.file_paths, self.template_data,
                progress_callback=lambda pct, fname: self.progress_update.emit(pct, fname),
                cancel_check=lambda: self._cancelled,
            )
        except Exception as exc:
            errors = [f"Validation failed: {exc}"]
        self.finished.emit(errors)
