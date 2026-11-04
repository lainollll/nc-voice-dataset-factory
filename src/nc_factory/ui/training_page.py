"""Guided LFM3 controls; all inspection and training run on owned workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from ..quick_setup import APPLICATION_ROOT
from ..training import TrainingRequest
from ..training_notebook import NOTEBOOK_URL
from .sections import CollapsibleSection
from .training_controller import TrainingController
from .training_data import TrainingSourceSummary, inspect_training_source
from .workers import TaskRunner


def _note(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label


class TrainingPageMixin:
    def _training_page(self) -> None:
        layout = self.page("LFM3 Fine-tuning", "Train a voice from your accepted recordings, one step at a time.")
        self.training_inspector = TaskRunner(self)
        self.training_inspector.result.connect(self._training_source_ready)
        self.training_inspector.error.connect(self._training_source_error)
        self.training_inspector.finished.connect(self._training_inspection_finished)
        self.training_source_timer = QTimer(self)
        self.training_source_timer.setSingleShot(True)
        self.training_source_timer.setInterval(180)
        self.training_source_timer.timeout.connect(self._refresh_training_source)
        self._training_source_version = 0
        self._training_source_pending = False
        self._training_source_valid = False
        self._training_has_activity = False
        self._training_operation = ""
        self.training_controller = TrainingController(
            APPLICATION_ROOT / "lfm3-training",
            lambda: not (self.runner.busy or self.aux_runner.busy or self.training_inspector.busy or self._closing),
            self,
        )

        data = QGroupBox("1. Choose your recordings")
        data_layout = QVBoxLayout(data)
        self.training_source = QComboBox()
        self.training_source.addItem("Current project", "project")
        self.training_source.addItem("Saved dataset export", "export")
        self.training_source.setAccessibleName("Training recording source")
        data_layout.addWidget(self.training_source)
        self.training_export_row = QWidget()
        self.training_export_row.setObjectName("trainingExportRow")
        export_layout = QHBoxLayout(self.training_export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        self.training_dataset = QLineEdit()
        self.training_dataset.setPlaceholderText("Choose a saved NC dataset folder")
        self.training_dataset.setAccessibleName("Saved dataset export folder")
        self.training_choose = self.button("Choose export…", self._choose_training_export)
        export_layout.addWidget(self.training_dataset, 1)
        export_layout.addWidget(self.training_choose)
        data_layout.addWidget(self.training_export_row)
        self.training_source_info = _note()
        data_layout.addWidget(self.training_source_info)
        source_actions = QVBoxLayout()
        source_actions.addWidget(self.button("Open project…", self.choose_project))
        self.training_current = self.button("Use current project", lambda: self.training_source.setCurrentIndex(0))
        source_actions.addWidget(self.training_current)
        data_layout.addLayout(source_actions)
        preview = self.button("Open Chatterbox preview", lambda: self.open_page("Voice Manager"))
        preview.setToolTip("Opens Voice Manager to choose and preview a reference WAV. This does not preview a trained LFM3 adapter.")
        data_layout.addWidget(preview)
        layout.addWidget(data)

        tools = QGroupBox("2. Prepare training")
        tools_layout = QVBoxLayout(tools)
        self.training_environment = _note()
        tools_layout.addWidget(self.training_environment)
        self.training_setup = self.button("Set up training environment", self.training_controller.setup)
        tools_layout.addWidget(self.training_setup)
        tools_layout.addWidget(_note(
            "Setup installs separate training tools. The first training run downloads the LFM3 model. "
            "Training needs a compatible NVIDIA GPU; package setup alone does not check GPU capacity."
        ))
        layout.addWidget(tools)

        run = QGroupBox("3. Train and save your voice")
        run_layout = QVBoxLayout(run)
        self.training_length = _note("Start with a short run: 200 training steps. Adjust this under Advanced.")
        run_layout.addWidget(self.training_length)
        self.training_start = self.button("Start fine-tuning", self._start_training, True)
        run_layout.addWidget(self.training_start)
        self.training_action_hint = _note()
        run_layout.addWidget(self.training_action_hint)
        self.training_stage = _note("Not started")
        self.training_stage.hide()
        run_layout.addWidget(self.training_stage)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.hide()
        run_layout.addWidget(self.training_progress)
        self.training_status = _note()
        self.training_status.hide()
        run_layout.addWidget(self.training_status)
        self.training_stop = self.button("Stop and save", self.training_controller.stop)
        self.training_stop.setToolTip("Request a safe stop. Loading or encoding finishes its current stage first.")
        run_layout.addWidget(self.training_stop)
        self.training_saved_files = _note()
        self.training_saved_files.hide()
        run_layout.addWidget(self.training_saved_files)
        self.training_adapter = self.button("Open saved voice files", self._open_training_adapter)
        self.training_adapter.setEnabled(False)
        self.training_adapter.hide()
        run_layout.addWidget(self.training_adapter)
        layout.addWidget(run)

        self.training_advanced = CollapsibleSection("Advanced training options")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.training_voice = QLineEdit()
        self.training_voice.setPlaceholderText("Use the dataset's voice name")
        form.addRow("Voice name override", self.training_voice)
        self.training_steps = QSpinBox()
        self.training_steps.setRange(1, 1_000_000)
        self.training_steps.setValue(200)
        self.training_steps.valueChanged.connect(lambda value: self.training_length.setText(
            f"Training length: {value:,} steps. A short run tests the workflow; voice quality is not guaranteed."
        ))
        form.addRow("Training steps", self.training_steps)
        self.training_advanced.body.addLayout(form)
        self.training_advanced.body.addWidget(_note(
            "The result is an LFM3 voice adapter, saved separately from Chatterbox. "
            "It is not automatically activated in NeuralCompanion. Swedish training remains experimental."
        ))
        layout.addWidget(self.training_advanced)

        self.training_details = CollapsibleSection("Technical details and logs")
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setAccessibleName("Training log")
        self.training_log.document().setMaximumBlockCount(500)
        self.training_log.setMinimumHeight(150)
        self.training_details.body.addWidget(self.training_log)
        self.training_output = self.button("Open run folder / checkpoints", self._open_training_output)
        self.training_details.body.addWidget(self.training_output)
        self.training_force = self.button("Force stop…", self._confirm_force_stop)
        self.training_force.setToolTip("Emergency stop: unsaved work and the current checkpoint may be lost.")
        self.training_details.body.addWidget(self.training_force)
        self.training_details.body.addWidget(self.button(
            "View original notebook", lambda: QDesktopServices.openUrl(QUrl(NOTEBOOK_URL))
        ))
        layout.addWidget(self.training_details)
        layout.addStretch()

        self.training_source.currentIndexChanged.connect(self._training_source_changed)
        self.training_dataset.textChanged.connect(self._training_export_edited)
        self.training_controller.log.connect(self.training_log.appendPlainText)
        self.training_controller.status.connect(self._training_status_message)
        self.training_controller.event.connect(self._training_progress)
        self.training_controller.busy_changed.connect(self._training_busy_changed)
        self.training_controller.finished.connect(self._training_finished)
        self.runner.finished.connect(self._training_background_finished)
        self.aux_runner.finished.connect(self._training_background_finished)
        self._training_source_changed()

    @property
    def training_busy(self) -> bool:
        return hasattr(self, "training_controller") and self.training_controller.busy

    def _training_source_key(self) -> tuple[int, str]:
        return (id(self.repo), self.training_dataset.text().strip()
                if self.training_source.currentData() == "export" else "")

    def _training_source_changed(self, *_args: Any) -> None:
        self._training_source_version += 1
        self._training_source_valid = False
        self._training_source_pending = True
        self.training_export_row.setVisible(self.training_source.currentData() == "export")
        self.training_current.setVisible(self.training_source.currentData() == "export")
        self.training_source_info.setText("Checking the selected recordings…")
        if not self.training_busy:
            self._training_has_activity = False
        self.training_source_timer.start()
        self._update_training_controls()

    def _training_export_edited(self, text: str) -> None:
        if text.strip() and self.training_source.currentData() != "export":
            self.training_source.setCurrentIndex(1)
        self._training_source_changed()

    def _refresh_training_source(self) -> None:
        self.training_source_timer.stop()
        if self._closing:
            return
        if self.training_busy or self.runner.busy or self.aux_runner.busy:
            self._training_source_pending = True
            self._update_training_controls()
            return
        if self.training_inspector.busy:
            self._training_source_pending = True
            return
        self._training_source_pending = False
        self._training_source_valid = False
        key, version = self._training_source_key(), self._training_source_version
        repo = self.repo
        if self.training_source.currentData() == "export":
            value = self.training_dataset.text().strip()
            if not value:
                self.training_source_info.setText("Choose a saved dataset export folder to continue.")
                self._update_training_controls()
                return
            export = Path(value)
        else:
            export = None
            if repo is None:
                self.training_source_info.setText("Open a project with accepted recordings, or choose a saved dataset.")
                self._update_training_controls()
                return
        self.training_source_info.setText("Checking the selected recordings…")
        self.training_inspector.start(lambda: (version, key, inspect_training_source(repo, export)))
        self._update_training_controls()

    @Slot(object)
    def _training_source_ready(self, result: tuple[int, tuple[int, str], TrainingSourceSummary]) -> None:
        version, key, summary = result
        if self._closing or version != self._training_source_version or key != self._training_source_key():
            return
        self._training_source_valid = summary.ready
        self.training_source_info.setText(summary.message + (
            f"\n{summary.voice} · {summary.language.upper()} · {summary.accepted:,} accepted recordings · "
            f"{summary.seconds / 60:.1f} minutes" if summary.ready else ""
        ))

    @Slot(str)
    def _training_source_error(self, message: str) -> None:
        if not self._training_source_pending:
            self._training_source_valid = False
            self.training_source_info.setText("Unable to inspect recordings. " + message.strip().splitlines()[-1])

    @Slot()
    def _training_inspection_finished(self) -> None:
        if self._closing:
            self.close()
        elif self._training_source_pending:
            self._refresh_training_source()
        else:
            self._update_training_controls()

    @Slot()
    def _training_background_finished(self) -> None:
        if self._closing:
            return
        if (self.navigation.currentItem() and self.navigation.currentItem().text() == "LFM3 Fine-tuning"
                and (self._training_source_pending or self.sender() is self.runner)):
            self._training_source_changed()
        else:
            self._update_training_controls()

    def _update_training_controls(self) -> None:
        busy = self.training_busy
        waiting = self.runner.busy or self.aux_runner.busy or self.training_inspector.busy
        idle = not (busy or waiting or self._closing)
        ready = self.training_controller.ready
        self.training_environment.setText(
            "Preparing training tools…" if busy and self.training_controller.stage == "setup"
            else "Ready · Separate training tools are installed." if ready
            else "Set up the separate training environment before starting."
        )
        self.training_setup.setText("Repair training environment" if ready else "Set up training environment")
        self.training_setup.setEnabled(idle)
        self.training_start.setEnabled(idle and ready and self._training_source_valid
                                       and not self.training_source_timer.isActive())
        for widget in (self.training_source, self.training_dataset, self.training_choose,
                       self.training_current, self.training_voice, self.training_steps):
            widget.setEnabled(not busy and not self._closing)
        self.training_stop.setEnabled(busy)
        self.training_stop.setVisible(busy)
        self.training_force.setEnabled(busy)
        self.training_output.setEnabled(self.training_controller.run_root is not None)
        reason = ("Training is active. Stop and save before changing tasks." if busy
                  else "Wait for the current background operation to finish." if waiting
                  else "Choose recordings in step 1 to continue." if not self._training_source_valid
                  else "Complete setup in step 2 to continue." if not ready
                  else "Ready to start. Source audio is rechecked before training.")
        self.training_action_hint.setText(reason)
        if not self._training_has_activity:
            self.training_status.setText("")
            self.training_status.hide()

    @Slot(str)
    def _training_status_message(self, message: str) -> None:
        self.training_status.setText(message)
        self.training_status.setVisible(bool(message))

    @Slot()
    def _choose_training_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose an NC dataset export")
        if path:
            self.training_source.setCurrentIndex(1)
            self.training_dataset.setText(path)

    @Slot()
    def _start_training(self) -> None:
        self._update_training_controls()
        if not self.training_start.isEnabled():
            return
        path = self.training_dataset.text().strip() if self.training_source.currentData() == "export" else ""
        self.training_controller.start_training(TrainingRequest(
            dataset_root=Path(path) if path else None,
            project_root=self.repo.root if not path and self.repo is not None else None,
            voice_name=self.training_voice.text(), max_steps=self.training_steps.value(),
        ))

    @Slot(bool)
    def _training_busy_changed(self, busy: bool) -> None:
        if busy:
            self._training_has_activity = True
            self._training_operation = self.training_controller.stage
            self.training_stage.setText("Preparing training tools…" if self._training_operation == "setup"
                                        else "Preparing your recordings…")
            self.training_progress.show()
            self.training_progress.setRange(0, 0)
            self.training_stage.show()
            self.training_adapter.hide()
            self.training_saved_files.hide()
        self._update_training_controls()
        self._update_voice_preview_controls()

    @Slot(object)
    def _training_progress(self, event: dict[str, Any]) -> None:
        stage = event.get("stage", "")
        self.training_stage.show()
        self.training_stage.setText({
            "model": "Loading the LFM3 model and codec…", "dataset": "Checking your recordings…",
            "encoding": "Encoding your recordings…", "saving": "Saving your voice files…",
            "completed": "Voice files saved",
        }.get(stage, f"Training step {event.get('step', 0)} / {event.get('total', '?')}"))
        self.training_progress.show()
        if stage == "training" and event.get("total"):
            self.training_progress.setRange(0, event["total"])
            self.training_progress.setValue(event.get("step", 0))
        else:
            self.training_progress.setRange(0, 0)
        if event.get("message"):
            self._training_status_message(event["message"])
        if event.get("metrics"):
            self.training_log.appendPlainText(str(event["metrics"]))

    @Slot(int)
    def _training_finished(self, code: int) -> None:
        self.training_stage.show()
        if self._training_operation == "setup" and code == 0:
            self.training_stage.setText("Training tools ready")
            self.training_progress.hide()
        elif code == 0:
            self.training_stage.setText("Training completed · voice files saved")
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(100)
        else:
            self.training_stage.setText("Stopped" if code == 130 else "Training needs attention")
            self.training_progress.hide()
            self.training_details.setExpanded(True)
        root = self.training_controller.run_root
        adapter = root / "artifacts" / "adapter" if root else None
        saved = bool(adapter and (adapter / "adapter_config.json").is_file()
                     and (adapter / "adapter_model.safetensors").is_file())
        self.training_adapter.setEnabled(saved)
        self.training_adapter.setVisible(saved)
        self.training_saved_files.setText(f"Saved LFM3 voice adapter: {adapter}" if saved else "")
        self.training_saved_files.setVisible(saved)
        self._update_training_controls()
        if self._closing:
            self.close()
        elif self._training_source_pending:
            self.training_source_timer.start()

    @Slot()
    def _confirm_force_stop(self) -> None:
        if self.training_busy and QMessageBox.question(
            self, "Force stop training?", "Unsaved work and the current checkpoint may be lost.\n"
            "Use Stop and save when possible. Force stop now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self.training_controller.force_stop()

    @Slot()
    def _open_training_output(self) -> None:
        if self.training_controller.run_root is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.training_controller.run_root)))

    @Slot()
    def _open_training_adapter(self) -> None:
        if self.training_controller.run_root is not None and self.training_adapter.isEnabled():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.training_controller.run_root / "artifacts" / "adapter")))
