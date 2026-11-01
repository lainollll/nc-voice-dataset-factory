from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot


class Worker(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, task: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            value = self._task()
        except Exception:
            self.error.emit(traceback.format_exc())
        else:
            self.result.emit(value)
        finally:
            self.finished.emit()


class TaskRunner(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(self, task: Callable[[], Any]) -> None:
        if self._thread is not None:
            raise RuntimeError("A background task is already active")

        thread = QThread(self)
        worker = Worker(task)
        worker.moveToThread(thread)

        worker.result.connect(self.result)
        worker.error.connect(self.error)
        worker.finished.connect(thread.quit)

        thread.started.connect(worker.run)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self._thread
        if thread is None:
            return
        # finished can precede native/TLS cleanup. Keep both wrappers owned until
        # the thread has actually joined, without blocking the GUI event loop.
        if not thread.wait(0):
            QTimer.singleShot(5, self._on_thread_finished)
            return
        self._thread = None
        self._worker = None
        thread.deleteLater()
        self.finished.emit()

    def close(self) -> None:
        if self.busy:
            raise RuntimeError("Wait for the background task to finish before closing")
