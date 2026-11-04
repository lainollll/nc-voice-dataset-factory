"""Own isolated setup/training processes without blocking the Qt event loop."""

from __future__ import annotations

import codecs
import json
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal, Slot

from ..storage import atomic_json
from ..training import PreparedTraining, TrainingRequest, prepare_training
from .workers import TaskRunner


def training_python(root: Path) -> Path:
    return root / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment_commands(root: Path) -> list[tuple[str, list[str]]]:
    python = training_python(root)
    commands = []
    if not python.is_file():
        commands.append(
            (str(getattr(sys, "_base_executable", sys.executable)), ["-m", "venv", str(root / "env")])
        )
    commands.extend(
        [
            (
                str(python),
                [
                    "-m",
                    "pip",
                    "install",
                    "torch==2.9.1",
                    "torchaudio==2.9.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
            ),
            (
                str(python),
                [
                    "-m",
                    "pip",
                    "install",
                    "transformers==5.20.2",
                    "datasets==3.6.0",
                    "pyarrow==20.0.0",
                    "accelerate",
                    "peft",
                    "librosa",
                    "soundfile",
                    "numpy==1.26.4",
                ],
            ),
            (
                str(python),
                [
                    "-c",
                    "import torch, peft, numpy as np, pyarrow as pa; "
                    "from transformers import AutoModelForMultimodalLM, AutoProcessor; "
                    "from datasets import Dataset, DatasetDict; "
                    "assert pa.array(np.array([1, 2], dtype=np.int64)).to_pylist() == [1, 2]; "
                    "assert Dataset.from_dict({'text': ['check']})[0]['text'] == 'check'; "
                    "print(f'LFM3 environment verified (NumPy {np.__version__}, PyArrow {pa.__version__})')",
                ],
            ),
        ]
    )
    return commands


class TrainingController(QObject):
    log = Signal(str)
    status = Signal(str)
    event = Signal(object)
    busy_changed = Signal(bool)
    finished = Signal(int)

    def __init__(self, root: Path, can_start: Callable[[], bool], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.root, self.can_start = root, can_start
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.worker = TaskRunner(self)
        self.worker.result.connect(self._prepared_ready)
        self.worker.error.connect(self._prepare_error)
        self.worker.finished.connect(self._prepare_finished)
        self._cancel = threading.Event()
        self._commands: list[tuple[str, list[str]]] = []
        self._prepared: PreparedTraining | None = None
        self._stage = ""
        self._text = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._killer: QProcess | None = None

    @property
    def busy(self) -> bool:
        return (
            self.worker.busy or self.process.state() != QProcess.ProcessState.NotRunning or bool(self._stage)
        )

    @property
    def ready(self) -> bool:
        return training_python(self.root).is_file() and (self.root / "environment_ready.json").is_file()

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def run_root(self) -> Path | None:
        return self._prepared.run_root if self._prepared else None

    def _begin(self, stage: str) -> bool:
        if self.busy or not self.can_start():
            self.status.emit("Another operation is active. Stop it or wait before fine-tuning.")
            return False
        self._stage = stage
        self._cancel.clear()
        self.busy_changed.emit(True)
        return True

    @Slot()
    def setup(self) -> None:
        if not self._begin("setup"):
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if (self.root / "env").resolve() == Path(sys.prefix).resolve():
                raise ValueError("Training must use its own environment")
            (self.root / "environment_ready.json").unlink(missing_ok=True)
            self._commands = environment_commands(self.root)
            self.status.emit(
                "Installing the separate LFM3 environment. Package downloads can take several minutes."
            )
            self._next_command()
        except Exception as exc:
            self.status.emit(str(exc))
            self._end(1)

    def start_training(self, request: TrainingRequest) -> None:
        if not self.ready:
            self.status.emit("Click Set up training environment first.")
            return
        if not self._begin("prepare"):
            return
        self._prepared = None
        root, cancel = self.root, self._cancel
        self.status.emit("Preparing a validated dataset snapshot and the official notebook…")
        self.worker.start(lambda: prepare_training(request, root, cancel))

    @Slot(object)
    def _prepared_ready(self, prepared: PreparedTraining) -> None:
        self._prepared = prepared

    @Slot(str)
    def _prepare_error(self, message: str) -> None:
        self.log.emit(message)
        self.status.emit(message.strip().splitlines()[-1])

    @Slot()
    def _prepare_finished(self) -> None:
        if self._cancel.is_set() or self._prepared is None:
            self._end(130 if self._cancel.is_set() else 1)
            return
        self._stage = "training"
        self.status.emit("Loading LFM3 and encoding your recordings. Follow the log below.")
        self._launch(
            str(training_python(self.root)), ["-u", str(self._prepared.runner)], self._prepared.run_root
        )

    def _next_command(self) -> None:
        if self._cancel.is_set():
            self.status.emit("Environment setup stopped.")
            self._end(130)
        elif self._commands:
            program, arguments = self._commands.pop(0)
            self._launch(program, arguments, self.root)
        else:
            atomic_json(
                self.root / "environment_ready.json", {"transformers": "5.20.2", "torch": "2.9.1+cu128"}
            )
            self.status.emit(
                "Training environment ready. Choose your dataset or use the current project, then Start."
            )
            self._end(0)

    def _launch(self, program: str, arguments: list[str], directory: Path) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._text = ""
        environment = QProcessEnvironment.systemEnvironment()
        environment.remove("PYTHONPATH")
        environment.remove("PYTHONHOME")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("VIRTUAL_ENV", str(self.root / "env"))
        environment.insert(
            "PATH", str(training_python(self.root).parent) + os.pathsep + environment.value("PATH")
        )
        environment.insert("HF_HOME", str(self.root / "model-cache"))
        environment.insert("HF_DATASETS_CACHE", str(self.root / "dataset-cache"))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(directory))
        self.process.start(program, arguments)

    @Slot()
    def _read_output(self) -> None:
        self._text += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self._text:
            line, self._text = self._text.split("\n", 1)
            self._line(line.rstrip("\r"))
        if len(self._text) > 65536:
            self._line(self._text[:65536])
            self._text = self._text[65536:]

    def _line(self, line: str) -> None:
        if line.startswith("NC_TRAINING_EVENT "):
            try:
                self.event.emit(json.loads(line.removeprefix("NC_TRAINING_EVENT ")))
            except ValueError:
                self.log.emit(line[:4000])
        else:
            self.log.emit(line[:4000])

    def _process_finished(self, code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._text:
            self._line(self._text)
            self._text = ""
        if self._stage == "setup" and not code and not self._cancel.is_set():
            try:
                self._next_command()
            except Exception as exc:
                self.status.emit(f"Environment setup failed: {exc}")
                self._end(1)
            return
        if self._cancel.is_set():
            self.status.emit("Stopped. Available checkpoints and logs remain in the run folder.")
            code = 130
        elif code:
            self.status.emit(f"{self._stage.capitalize()} failed (exit {code}). See the log below.")
        elif self._stage == "training" and self.run_root is not None:
            adapter = self.run_root / "artifacts" / "adapter"
            if (
                not (adapter / "adapter_config.json").is_file()
                or not (adapter / "adapter_model.safetensors").is_file()
            ):
                code = 1
                self.status.emit("Training ended without a saved adapter. Inspect the run log.")
            else:
                self.status.emit(f"Training completed. Adapter saved to {adapter}")
        self._end(code)

    @Slot(QProcess.ProcessError)
    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.status.emit("Unable to start the training process: " + self.process.errorString())
            self._end(1)

    def _end(self, code: int) -> None:
        self._commands.clear()
        self._stage = ""
        self.busy_changed.emit(False)
        self.finished.emit(code)

    @Slot()
    def stop(self) -> None:
        if not self.busy:
            return
        self._cancel.set()
        if self._stage == "training" and self.run_root is not None:
            try:
                (self.run_root / "STOP").touch()
            except OSError as exc:
                self.status.emit(f"Unable to request a saved stop: {exc}. Use Force stop if needed.")
                return
            self.status.emit(
                "Stop requested. Saving at the next training step; model loading finishes its current stage."
            )
        elif self.process.state() != QProcess.ProcessState.NotRunning:
            pid = self.process.processId()
            self.process.terminate()
            QTimer.singleShot(5000, lambda: self.force_stop() if self.process.processId() == pid else None)
        else:
            self.status.emit("Stopping after dataset preparation finishes…")

    @Slot()
    def force_stop(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.stop()
            return
        self._cancel.set()
        if os.name == "nt":
            self._killer = QProcess(self)
            self._killer.start("taskkill", ["/PID", str(self.process.processId()), "/T", "/F"])
        else:
            self.process.kill()
