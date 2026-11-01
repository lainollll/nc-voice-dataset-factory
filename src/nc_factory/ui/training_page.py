"""LFM3 fine-tuning controls, backed by an isolated external process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
)

from ..quick_setup import APPLICATION_ROOT
from ..training import TrainingRequest
from ..training_notebook import NOTEBOOK_URL
from .training_controller import TrainingController


class TrainingPageMixin:
    def _training_page(self) -> None:
        layout = self.page(
            "LFM3 Fine-tuning", "Train a LoRA voice adapter using the official Liquid notebook."
        )
        self.training_controller = TrainingController(
            APPLICATION_ROOT / "lfm3-training",
            lambda: not (self.runner.busy or self.aux_runner.busy or self._closing),
            self,
        )
        self.training_dataset = QLineEdit()
        self.training_dataset.setPlaceholderText("Current project's accepted recordings (automatic snapshot)")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Dataset export (optional)", self.training_dataset)
        dataset_buttons = QHBoxLayout()
        self.training_choose = self.button("Choose export…", self._choose_training_export)
        self.training_current = self.button("Use current project", self.training_dataset.clear)
        dataset_buttons.addWidget(self.training_choose)
        dataset_buttons.addWidget(self.training_current)
        form.addRow(dataset_buttons)
        self.training_voice = QLineEdit()
        self.training_voice.setPlaceholderText("Use the dataset's voice name")
        form.addRow("Voice name (optional)", self.training_voice)
        self.training_steps = QSpinBox()
        self.training_steps.setRange(1, 1_000_000)
        self.training_steps.setValue(200)
        form.addRow("Training steps", self.training_steps)
        layout.addLayout(form)
        note = QLabel(
            "Set up the training environment once, then Start. Your current project's accepted audio "
            "is exported automatically if no export folder is selected. The first run downloads LFM3 "
            "model weights. Training uses a separate environment and GPU; speech generation waits. "
            "The official example recommends an A100; this setup uses smaller batches. "
            "Swedish fine-tuning remains experimental."
        )
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        layout.addWidget(note)
        self.training_setup = self.button("Set up training environment", self.training_controller.setup)
        self.training_start = self.button("Start fine-tuning", self._start_training, True)
        layout.addWidget(self.training_setup)
        layout.addWidget(self.training_start)
        controls = QHBoxLayout()
        self.training_stop = self.button("Stop and save", self.training_controller.stop)
        self.training_force = self.button("Force stop", self.training_controller.force_stop)
        self.training_force.setToolTip(
            "Terminate the process; unsaved work and the current checkpoint may be lost"
        )
        controls.addWidget(self.training_stop)
        controls.addWidget(self.training_force)
        layout.addLayout(controls)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        layout.addWidget(self.training_progress)
        self.training_status = QLabel(
            "Set up the separate environment, then Start with your current dataset."
        )
        self.training_status.setWordWrap(True)
        self.training_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.training_status)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.document().setMaximumBlockCount(500)
        self.training_log.setMinimumHeight(150)
        layout.addWidget(self.training_log)
        self.training_output = self.button("Open run / checkpoints", self._open_training_output)
        layout.addWidget(self.training_output)
        layout.addWidget(
            self.button("View original notebook", lambda: QDesktopServices.openUrl(QUrl(NOTEBOOK_URL)))
        )
        self.training_controller.log.connect(self.training_log.appendPlainText)
        self.training_controller.status.connect(self.training_status.setText)
        self.training_controller.event.connect(self._training_progress)
        self.training_controller.busy_changed.connect(self._training_busy_changed)
        self.training_controller.finished.connect(self._training_finished)
        self._training_busy_changed(False)

    @property
    def training_busy(self) -> bool:
        return hasattr(self, "training_controller") and self.training_controller.busy

    @Slot()
    def _choose_training_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose an NC dataset export")
        if path:
            self.training_dataset.setText(path)

    @Slot()
    def _start_training(self) -> None:
        path = self.training_dataset.text().strip()
        self.training_controller.start_training(
            TrainingRequest(
                dataset_root=Path(path) if path else None,
                project_root=self.repo.root if not path and self.repo is not None else None,
                voice_name=self.training_voice.text(),
                max_steps=self.training_steps.value(),
            )
        )

    @Slot(bool)
    def _training_busy_changed(self, busy: bool) -> None:
        if not busy:
            self.training_setup.setText(
                "Repair training environment"
                if self.training_controller.ready
                else "Set up training environment"
            )
        for widget in (
            self.training_setup,
            self.training_start,
            self.training_dataset,
            self.training_choose,
            self.training_current,
            self.training_voice,
            self.training_steps,
        ):
            widget.setEnabled(not busy)
        self.training_stop.setEnabled(busy)
        self.training_force.setEnabled(busy)
        self.training_output.setEnabled(self.training_controller.run_root is not None)
        if busy:
            self.training_progress.setRange(0, 0)
        elif not self.training_progress.maximum():
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(0)

    @Slot(object)
    def _training_progress(self, event: dict[str, Any]) -> None:
        if event.get("total"):
            self.training_progress.setRange(0, event["total"])
            self.training_progress.setValue(event.get("step", 0))
        if event.get("message"):
            self.training_status.setText(event["message"])
        elif event.get("stage") == "training":
            self.training_status.setText(f"Training step {event.get('step', 0)} / {event.get('total', '?')}")
        if event.get("metrics"):
            self.training_log.appendPlainText(str(event["metrics"]))

    @Slot(int)
    def _training_finished(self, _code: int) -> None:
        if self._closing:
            self.close()

    @Slot()
    def _open_training_output(self) -> None:
        path = self.training_controller.run_root
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
