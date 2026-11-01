"""Simple setup UI; all discovery and preparation run through the existing workers."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable
import uuid

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..models import download_model, model_download_plan
from ..quick_setup import (
    APPLICATION_ROOT,
    PreparedDataset,
    QuickStartRequest,
    discover_models,
    prepare_dataset,
)
from ..text import diversity_statistics


class QuickStartPage(QWidget):
    generate_requested = Signal(object)
    models_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    export_requested = Signal()
    train_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.reference = QLineEdit()
        self.reference.setReadOnly(True)
        self.reference.setPlaceholderText("Choose the voice recording to use")
        self.wav_button = QPushButton("Choose WAV…")
        self.wav_button.clicked.connect(self._choose_wav)
        wav_row = QHBoxLayout()
        wav_row.addWidget(self.reference, 1)
        wav_row.addWidget(self.wav_button)
        form.addRow("Voice WAV", wav_row)
        self.language = QComboBox()
        self.language.addItem("English", "en")
        self.language.addItem("Swedish", "sv")
        form.addRow("Dataset language", self.language)
        self.size = QSpinBox()
        self.size.setRange(1, 1_000_000)
        self.size.setValue(1000)
        self.size_unit = QComboBox()
        self.size_unit.addItem("recordings", "count")
        self.size_unit.addItem("minutes of audio", "minutes")
        self.size_unit.currentIndexChanged.connect(self._unit_changed)
        size_row = QHBoxLayout()
        size_row.addWidget(self.size, 1)
        size_row.addWidget(self.size_unit)
        form.addRow("Dataset size", size_row)
        layout.addLayout(form)
        self.permission = QCheckBox("I own or have permission to use this voice")
        self.reference.textChanged.connect(lambda: self.permission.setChecked(False))
        layout.addWidget(self.permission)
        defaults = QLabel(
            "Models, sentences, GPU workers and quality checks are set up automatically. "
            "Only accepted recordings count toward your size. "
            "Each run creates a new project in this app's projects folder. "
            "Use the other cards only when you want to adjust advanced settings."
        )
        defaults.setWordWrap(True)
        defaults.setObjectName("subtitle")
        layout.addWidget(defaults)
        self.generate_button = QPushButton("Generate dataset")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)
        self.models_button = QPushButton("Inspect missing models")
        self.models_button.clicked.connect(self.models_requested.emit)
        layout.addWidget(self.models_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("Choose your WAV and dataset size, then click Generate dataset.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.output = QLabel()
        self.output.setWordWrap(True)
        self.output.setObjectName("subtitle")
        layout.addWidget(self.output)
        controls = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_requested.emit)
        controls.addWidget(self.pause_button)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        controls.addWidget(self.resume_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        self.export_button = QPushButton("Export accepted dataset…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_button)
        training_button = QPushButton("Fine-tune with LFM3")
        training_button.clicked.connect(self.train_requested.emit)
        layout.addWidget(training_button)
        self._busy = self._running = self._can_resume = self._can_export = False
        self._refresh_controls()

    @Slot()
    def _choose_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose voice recording", filter="WAV audio (*.wav)")
        if path:
            self.reference.setText(path)

    @Slot()
    def _unit_changed(self) -> None:
        minutes = self.size_unit.currentData() == "minutes"
        self.size.setMaximum(100_000 if minutes else 1_000_000)
        self.size.setValue(60 if minutes else 1000)

    @Slot()
    def _generate(self) -> None:
        if not self.reference.text().strip():
            self.status.setText("Choose your voice WAV first.")
            return
        minutes = self.size_unit.currentData() == "minutes"
        self.generate_requested.emit(
            QuickStartRequest(
                Path(self.reference.text()),
                self.language.currentData(),
                0 if minutes else self.size.value(),
                self.size.value() if minutes else 0,
                self.permission.isChecked(),
            )
        )

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (
            self.wav_button,
            self.language,
            self.size,
            self.size_unit,
            self.permission,
            self.generate_button,
            self.models_button,
        ):
            widget.setEnabled(not busy)
        self._refresh_controls()

    def set_running(self, running: bool) -> None:
        self._running = running
        self._refresh_controls()

    def set_result(self, accepted: int, resumable: bool) -> None:
        self._can_export, self._can_resume = accepted > 0, resumable
        self._running = False
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        self.pause_button.setEnabled(self._busy and self._running)
        self.resume_button.setEnabled((self._busy and self._running) or (not self._busy and self._can_resume))
        self.stop_button.setEnabled(self._busy)
        self.export_button.setEnabled(not self._busy and self._can_export)


class QuickStartMixin:
    def _quick_start_page(self, layout: QVBoxLayout) -> None:
        self._quick_app_root = APPLICATION_ROOT
        self._quick_next: Callable[[], None] | None = None
        self._quick_prepared: PreparedDataset | None = None
        self._quick_active = False
        self._quick_generating = False
        self._quick_runtime_workers: int | None = None
        self._quick_runtime_activity: dict[str, Any] = {}
        self._quick_job_id: str | None = None
        self._quick_plans: tuple[dict[str, Any], ...] = ()
        self.quick_start = QuickStartPage()
        layout.addWidget(self.quick_start)
        self.quick_start.generate_requested.connect(self._prepare_quick_dataset)
        self.quick_start.models_requested.connect(self._quick_models)
        self.quick_start.language.currentIndexChanged.connect(self._clear_quick_plans)
        self.quick_start.pause_requested.connect(self.pause_job)
        self.quick_start.resume_requested.connect(self._quick_resume)
        self.quick_start.stop_requested.connect(self.stop_job)
        self.quick_start.export_requested.connect(self._quick_export)
        self.quick_start.train_requested.connect(lambda: self.open_page("LFM3 Fine-tuning"))
        self.runner.finished.connect(self._continue_quick_start)
        self.aux_runner.finished.connect(self._continue_quick_start)
        self.runner.error.connect(self._quick_error)
        self.progress_event.connect(self._quick_progress)

    def _quick_operation(self, task: Callable, handler: Callable, message: str) -> None:
        if self.runner.busy or self.aux_runner.busy or self.training_busy or self._quick_next is not None:
            self.quick_start.status.setText(
                "An operation is active. Wait or stop it before starting a new dataset."
            )
            return
        self._quick_active = True
        self.quick_start.set_busy(True)
        self.quick_start.progress.setRange(0, 0)
        self.quick_start.status.setText(message)
        self.run_task(task, handler)

    @Slot(object)
    def _prepare_quick_dataset(self, request: QuickStartRequest) -> None:
        preferred = tuple(
            Path(value) for value in (self.model_path.text(), self.asr_path.text()) if value.strip()
        )
        root, cancel = self._quick_app_root, self._task_cancel
        self._quick_operation(
            lambda: prepare_dataset(request, root, preferred=preferred, cancel=cancel),
            self._quick_prepared_ready,
            "Checking your WAV, finding local models and preparing your dataset…",
        )

    def _quick_prepared_ready(self, prepared: PreparedDataset) -> None:
        self._quick_prepared = prepared
        self._quick_next = self._activate_quick_dataset

    def _activate_quick_dataset(self) -> None:
        prepared = self._quick_prepared
        if prepared is None:
            return
        self._browser_offset = 0
        self._selected_sample = None
        self.pipeline = None
        self._quick_job_id = None
        self._project_opened(
            (
                prepared.repo,
                prepared.config,
                {"interrupted_reset": 0},
                [],
                [asdict(prepared.config.voice)],
                {},
            )
        )
        self.quick_start.output.setText(f"Project: {prepared.repo.root}")
        self.quick_start.set_result(0, False)
        self._quick_next = self._launch_quick_dataset

    @Slot()
    def _continue_quick_start(self) -> None:
        # A result arrives before its QThread has finished. Wait for both existing
        # runners and queued diagnostics before asking Batch Generator to start.
        if self.runner.busy or self.aux_runner.busy or self._aux_pending is not None:
            return
        if self._closing or self._task_cancel.is_set():
            self._quick_next = None
            if self._quick_active and not self._quick_generating:
                self.quick_start.status.setText("Operation cancelled.")
        if self._quick_next is not None:
            action, self._quick_next = self._quick_next, None
            try:
                action()
            except Exception as exc:
                self._quick_error(str(exc))
            if self.runner.busy or self.aux_runner.busy or self._quick_next is not None:
                return
        if self._quick_active:
            self._quick_active = self._quick_generating = False
            self.quick_start.set_running(False)
            self.quick_start.set_busy(False)
            if self.quick_start.progress.maximum() == 0:
                self.quick_start.progress.setRange(0, 100)
                self.quick_start.progress.setValue(0)

    def _launch_quick_dataset(self) -> None:
        prepared = self._quick_prepared
        if prepared is None:
            return
        self.apply_config(prepared.config)
        self._preview_ready((prepared.items, diversity_statistics(prepared.items), []))
        self._quick_generating = True
        self._quick_runtime_workers = None
        self._quick_runtime_activity = {}
        self.quick_start.set_running(True)
        self.quick_start.status.setText("Generating and checking recordings…")
        self.start_job()
        if self.runner.busy:
            self._result_handler = self._quick_job_finished
        else:
            self._quick_error(self.statusBar().currentMessage())

    def _quick_job_finished(self, result: dict[str, Any]) -> None:
        self._job_finished(result)
        self._quick_job_id = self.job_id
        accepted = result.get("accepted", 0)
        message = f"{result.get('status', 'Finished')}: {accepted} accepted recordings."
        if self._quick_runtime_workers is not None:
            message += f" GPU workers: {self._quick_runtime_workers}."
        if result.get("error"):
            message += " " + str(result["error"])
        self.quick_start.status.setText(message)
        self.quick_start.set_result(accepted, result.get("status") != "COMPLETED")

    @Slot()
    def _quick_resume(self) -> None:
        if self.runner.busy:
            if self._quick_generating:
                self.resume_job()
            return
        if self.aux_runner.busy or self.training_busy:
            self.quick_start.status.setText("Wait for the current operation to finish, then Resume.")
            return
        if (
            self._quick_prepared is None
            or self.repo is not self._quick_prepared.repo
            or self.job_id is None
            or self.job_id != self._quick_job_id
        ):
            self.quick_start.status.setText("Use Batch Generator to resume the selected project.")
            return
        self._quick_active = self._quick_generating = True
        self.quick_start.set_busy(True)
        self.quick_start.set_running(True)
        self.quick_start.status.setText("Resuming generation and quality checks…")
        self.resume_job()
        if self.runner.busy:
            self._result_handler = self._quick_job_finished
        else:
            self._continue_quick_start()

    @Slot(str)
    def _quick_error(self, message: str) -> None:
        if self._quick_active:
            self.quick_start.status.setText(
                message.strip().splitlines()[-1] if message.strip() else "Operation failed."
            )

    @Slot(object)
    def _quick_progress(self, event: dict[str, Any]) -> None:
        if not self._quick_generating or self._quick_prepared is None or "exported" in event:
            return
        config = self._quick_prepared.config
        if runtime := event.get("runtime"):
            self._quick_runtime_workers = runtime["workers"]
            self._quick_runtime_activity = runtime
        count, seconds = event.get("accepted", 0), event.get("accepted_seconds", 0)
        target = config.target_count or math.ceil(config.target_seconds)
        self.quick_start.progress.setRange(0, target)
        self.quick_start.progress.setValue(count if config.target_count else int(seconds))
        self.quick_start.status.setText(
            f"{count} accepted recordings · {seconds / 60:.1f} minutes · {event.get('status', 'Generating')}"
            + (
                f" · GPU workers: {self._quick_runtime_workers}"
                if self._quick_runtime_workers is not None
                else ""
            )
            + (
                f"\nActive GPU tasks: {self._quick_runtime_activity['generation_active']} · "
                f"Quality checks: {self._quick_runtime_activity['quality_active']}/"
                f"{self._quick_runtime_activity.get('quality_workers', 1)} · "
                f"Queued checks: {self._quick_runtime_activity['quality_waiting']}"
                if "generation_active" in self._quick_runtime_activity
                else ""
            )
            + (f"\n{event['error']}" if event.get("error") else "")
        )

    @Slot()
    def _clear_quick_plans(self) -> None:
        self._quick_plans = ()
        self.quick_start.models_button.setText("Inspect missing models")

    @Slot()
    def _quick_models(self) -> None:
        root, cancel = self._quick_app_root, self._task_cancel
        if self._quick_plans:
            plans = self._quick_plans

            def download() -> None:
                for plan in plans:
                    if cancel.is_set():
                        raise InterruptedError("Model download cancelled")
                    destination = root / "models" / f"{plan['kind']}-{uuid.uuid4().hex}"
                    download_model(plan, destination)

            self._quick_operation(download, self._quick_models_downloaded, "Downloading the selected models…")
            return
        language = self.quick_start.language.currentData()
        preferred = tuple(
            Path(value) for value in (self.model_path.text(), self.asr_path.text()) if value.strip()
        )

        def inspect() -> tuple[dict[str, Any], ...]:
            models = discover_models(root, language, preferred=preferred)
            return tuple(model_download_plan(kind) for kind in models.missing)

        self._quick_operation(
            inspect, self._quick_models_inspected, "Looking for missing models and download sizes…"
        )

    def _quick_models_inspected(self, plans: tuple[dict[str, Any], ...]) -> None:
        self._quick_plans = plans
        if not plans:
            self.quick_start.status.setText(
                "Matching speech and transcript models are installed. Click Generate dataset."
            )
            return
        size_gib = sum(plan["bytes"] for plan in plans) / 1024**3
        details = "\n".join(
            f"{plan['repository']} · {plan['bytes'] / 1024**3:.2f} GiB · License: {plan['license']}\n{plan['source']}"
            for plan in plans
        )
        self.quick_start.status.setText(
            "The following models will be saved in this app's models folder.\n"
            + details
            + "\nClick Download models when ready, then Generate dataset."
        )
        self.quick_start.models_button.setText(f"Download models ({size_gib:.2f} GiB)")

    def _quick_models_downloaded(self, _result: None) -> None:
        self._clear_quick_plans()
        self.quick_start.status.setText("Models installed. Click Generate dataset to continue.")

    @Slot()
    def _quick_export(self) -> None:
        self.open_page("Export")
        self.export_project()
