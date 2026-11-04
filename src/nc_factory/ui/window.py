from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from ..audio import sha256_file
from ..domain import ProjectConfig, QualitySettings, SynthesisSettings, TextItem, VoiceProfile
from ..pipeline import Pipeline
from ..storage import Repository
from .theme import DARK_STYLE, configure_fonts
from .workers import TaskRunner
from .guide import GettingStartedDock, GettingStartedGuide


from .voice_page import VoicePageMixin
from .text_page import TextPageMixin
from .batch_page import BatchPageMixin
from .quality_page import QualityPageMixin
from .browser_page import BrowserPageMixin
from .export_page import ExportPageMixin
from .quick_start import QuickStartMixin
from .training_page import TrainingPageMixin


class MainWindow(
    QMainWindow,
    VoicePageMixin,
    TextPageMixin,
    BatchPageMixin,
    QualityPageMixin,
    BrowserPageMixin,
    ExportPageMixin,
    QuickStartMixin,
    TrainingPageMixin,
):
    progress_event = Signal(object)

    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        configure_fonts()
        self.setWindowTitle("NC Voice Dataset Factory")
        self.resize(1250, 850)
        self.setStyleSheet(DARK_STYLE)
        self.repo: Repository | None = None
        self.pipeline: Pipeline | None = None
        self.job_id: str | None = None
        self._closing = False
        self._task_cancel = threading.Event()
        self._reference_digest = ""
        self._preview_items: list[TextItem] = []
        self._last_config: ProjectConfig | None = None
        self._browser_offset = 0
        self._browser_rows: list[dict[str, Any]] = []
        self._selected_sample: dict[str, Any] | None = None
        self._aux_handler: Callable | None = None
        self._aux_pending: tuple[Callable, Callable | None] | None = None
        self.aux_runner = TaskRunner(self)
        self.aux_runner.result.connect(self._aux_result)
        self.aux_runner.error.connect(self._error)
        self.aux_runner.finished.connect(self._aux_finished)
        self._result_handler: Callable[[Any], None] | None = None
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.error.connect(self._error)
        self.runner.finished.connect(self._finished)
        self.progress_event.connect(self._progress)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.7)
        body = QWidget()
        self.setCentralWidget(body)
        layout = QHBoxLayout(body)
        self.navigation = QListWidget()
        self.navigation.setObjectName("workspaceNavigation")
        self.navigation.setMaximumWidth(220)
        self.pages = QStackedWidget()
        layout.addWidget(self.navigation)
        layout.addWidget(self.pages, 1)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        guide_layout = self.page(
            "Getting Started",
            "Choose your voice and dataset size. The app handles the setup.",
        )
        self.guide = GettingStartedGuide(self)
        self.guide.navigate_requested.connect(self.open_page)
        self.guide.project_requested.connect(self.choose_project)
        self.guide.small_test_requested.connect(self.prepare_small_test)
        self.guide_dock = GettingStartedDock(self.guide, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.guide_dock)
        self.resizeDocks([self.guide_dock], [400], Qt.Orientation.Horizontal)
        self._quick_start_page(guide_layout)
        guide_layout.addWidget(self.button("Focus instructions", self.focus_guide))
        guide_layout.addStretch()
        self._dashboard()
        self._voice_page()
        self._text_page()
        self._batch_page()
        self._quality_page()
        self._browser_page()
        self._export_page()
        self._training_page()
        self.navigation.currentRowChanged.connect(self._page_changed)
        self.open_page("Dashboard" if project_root else "Getting Started")
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(15000)
        self.metrics_timer.timeout.connect(self.refresh_diagnostics)
        self.metrics_timer.start()
        self.statusBar().showMessage("Create or open a project to begin. All processing runs locally.")
        if project_root:
            QTimer.singleShot(0, lambda: self.open_project(project_root))

    def page(self, title: str, subtitle: str) -> QVBoxLayout:
        page = QWidget()
        page.setObjectName("workspacePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        heading = QLabel(title)
        heading.setWordWrap(True)
        heading.setObjectName("title")
        sub = QLabel(subtitle)
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(sub)
        self.navigation.addItem(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return layout

    @staticmethod
    def button(label: str, callback: Callable, primary: bool = False) -> QPushButton:
        button = QPushButton(label)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    @Slot(str)
    def open_page(self, title: str) -> None:
        for index in range(self.navigation.count()):
            if self.navigation.item(index).text() == title:
                self.navigation.setCurrentRow(index)
                return

    @Slot(int)
    def _page_changed(self, index: int) -> None:
        if self.navigation.item(index).text() == "LFM3 Fine-tuning":
            self.guide.show_training_help()
            self._refresh_training_source()
        elif self.navigation.item(index).text() == "Voice Manager":
            self.guide.show_voice_preview_help()

    @Slot()
    def focus_guide(self) -> None:
        self.guide_dock.show()
        self.guide_dock.raise_()
        if self.guide_dock.isFloating():
            self.guide_dock.activateWindow()
        self.guide.instructions.setFocus()

    @Slot()
    def prepare_small_test(self) -> None:
        if self.runner.busy or self.aux_runner.busy:
            self.statusBar().showMessage(
                "Wait for the active operation to finish before changing test targets."
            )
            return
        self.target_count.setValue(10)
        self.target_seconds.setValue(0)
        self.text_count.setValue(10)
        self.open_page("Batch Generator")
        self.statusBar().showMessage(
            "10-clip test prepared. Check your sentences, then click Start when ready."
        )

    def _dashboard(self) -> None:
        layout = self.page("Dashboard", "A reproducible voice-to-dataset workspace")
        layout.addWidget(self.button("Create / Open project…", self.choose_project, True))
        self.project_label = QLabel("No project selected")
        self.stats_label = QLabel("Accepted 0  •  Rejected 0  •  Duration 0 s")
        self.stats_label.setWordWrap(True)
        self.gpu_label = QLabel("GPU diagnostics have not been requested yet.")
        self.gpu_label.setWordWrap(True)
        self.storage_label = QLabel("Disk estimate appears after project configuration.")
        self.storage_label.setWordWrap(True)
        self.selected_voice_label = QLabel("Selected voice: none")
        layout.addWidget(self.project_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.selected_voice_label)
        layout.addWidget(self.gpu_label)
        layout.addWidget(self.storage_label)
        layout.addWidget(self.button("Refresh GPU / disk diagnostics", self.refresh_diagnostics))
        layout.addStretch()

    def refresh_diagnostics(self) -> None:
        if not self.repo or self._closing:
            return
        from ..diagnostics import diagnostics, estimate_storage

        try:
            cfg = self.configuration()
        except Exception:
            cfg = None
        root = self.repo.root

        def task():
            result = diagnostics(root)
            if cfg:
                result["storage_estimate"] = estimate_storage(cfg)
            return result

        self.run_aux(task, self._diagnostics_ready)

    def _diagnostics_ready(self, result: dict[str, Any]) -> None:
        gpu = result["gpu"]
        devices = gpu.get("nvidia_smi") or []
        if devices:
            device = devices[0]
            runtime = getattr(self, "_active_parallel_runtime", None)
            detail = (
                f"Generation: {runtime['device']} • {runtime['workers']} GPU worker(s) • ASR: {runtime['asr_device']}"
                if runtime and runtime["device"].startswith("cuda")
                else f"PyTorch allocated: {gpu['torch_cuda_allocated_bytes']} B • Reserved: {gpu['torch_cuda_reserved_bytes']} B • CUDA available: {gpu['cuda_available']}"
            )
            self.gpu_label.setText(
                f"{device['name']} • Free {device['memory_free_mb']} / {device['memory_total_mb']} MiB • Utilization {device['utilization_percent']}%\n{detail}"
            )
        else:
            self.gpu_label.setText(f"GPU: {gpu.get('reason') or 'unavailable'}")
        estimate = result.get("storage_estimate")
        free = result.get("disk_free_bytes")
        if estimate and free is not None:
            need = estimate["estimated_total_bytes"]
            self.storage_label.setText(
                f"Estimated working + export storage: {need / 1024**3:.2f} GiB • Free: {free / 1024**3:.2f} GiB\n{estimate['explanation']}"
                + ("\nWARNING: estimated storage exceeds available disk space." if need > free else "")
            )

    def run_aux(self, task: Callable, handler: Callable | None = None) -> None:
        if self.aux_runner.busy:
            self._aux_pending = (
                task,
                handler,
            )  # Latest read replaces one pending read: bounded backpressure.
            return
        self._aux_handler = handler
        self.aux_runner.start(task)
        if hasattr(self, "preview_button"):
            self._update_voice_preview_controls()
        if hasattr(self, "training_start"):
            self._update_training_controls()

    @Slot(object)
    def _aux_result(self, result: Any) -> None:
        if self._aux_handler:
            self._aux_handler(result)

    @Slot()
    def _aux_finished(self) -> None:
        if self._closing:
            self._aux_pending = None
            if not self.runner.busy:
                self.close()
        elif self._aux_pending:
            task, handler = self._aux_pending
            self._aux_pending = None
            self.run_aux(task, handler)

    def choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Create / Open project")
        if path:
            self.open_project(Path(path))

    def open_project(self, path: Path) -> None:
        def task():
            repo = Repository(path)
            from ..logging_config import configure_logging

            configure_logging(repo.root)
            with repo.writer_lock():
                recovery = repo.recover()
            try:
                config = repo.load_config()
            except ValueError:
                config = None
            return repo, config, recovery, repo.list_jobs(), repo.voices(), repo.statistics()

        self.run_task(task, self._project_opened)

    def _project_opened(self, result: tuple) -> None:
        self.repo, config, recovery, jobs, voices, stats = result
        self.project_label.setText(str(self.repo.root))
        if config:
            self.apply_config(config)
        self.saved_voices.clear()
        self.saved_voices.addItem("Select a saved project voice…", None)
        for voice in voices:
            self.saved_voices.addItem(voice["name"], voice)
        self.job_selector.clear()
        for job in jobs:
            self.job_selector.addItem(
                f"{job['created_at'][:19]} • {job['status']} • {job['id'][:8]}", job["id"]
            )
        self.job_id = jobs[0]["id"] if jobs else None
        self.refresh_stats()
        self.refresh_browser()
        self.refresh_diagnostics()
        self.statusBar().showMessage(
            f"Project opened. Recovered {recovery['interrupted_reset']} interrupted samples."
        )

    def apply_config(self, config: ProjectConfig) -> None:
        self._last_config = config
        self.voice_name.setText(config.voice.name)
        self.reference_path.setText(config.voice.reference_path)
        self._reference_digest = config.voice.reference_sha256
        self.permission.setChecked(config.voice.permission_confirmed)
        self.language.setCurrentIndex(self.language.findData(config.language))
        self.engine.setCurrentIndex(self.engine.findData(config.synthesis.engine))
        self.model_path.setText(config.synthesis.model_dir)
        self.service_config.setText(config.synthesis.service_config)
        self.service_voice_id.setText(config.voice.service_voice_id)
        self.target_count.setValue(config.target_count)
        self.target_seconds.setValue(config.target_seconds)
        self.max_retries.setValue(config.max_retries)
        self.max_attempts.setValue(config.max_attempts)
        self.minimum_free.setValue(config.minimum_free_gb)
        self.minimum_gpu_free.setValue(config.minimum_gpu_free_gb)
        self.audio_rate.setCurrentIndex(self.audio_rate.findData(config.audio_sample_rate))
        self.audio_subtype.setCurrentText(config.audio_subtype)
        self.generation_workers.setValue(config.generation_workers)
        self.auto_generation_workers.setChecked(config.auto_generation_workers)
        self.overlap_quality.setChecked(config.overlap_quality)
        self.require_cuda.setChecked(config.require_cuda)
        self.worker_vram.setValue(config.gpu_memory_per_worker_gb)
        self.device.setCurrentText(config.synthesis.device)
        self.random_seed.setValue(config.synthesis.seed)
        for key, spin in self.synthesis_spins.items():
            spin.setValue(getattr(config.synthesis, key))
        self.selected_voice_label.setText(
            f"Selected voice: {config.voice.name} • {config.voice.language} • {config.synthesis.engine}\nReference SHA-256: {config.voice.reference_sha256}"
        )
        self.text_method.setCurrentIndex(self.text_method.findData(config.text_method))
        self.min_words.setValue(config.min_words)
        self.max_words.setValue(config.max_words)
        self.topics.setText(", ".join(config.custom_topics))
        self.llm_endpoint.setText(config.llm_endpoint)
        self.llm_model.setCurrentText(config.llm_model)
        self.allow_remote.setChecked(config.allow_remote_llm)
        for key, value in config.category_weights.items():
            if key in self.category_weights:
                self.category_weights[key].setValue(value)
        self.verify_asr.setChecked(config.quality.verify_asr)
        self.asr_path.setText(config.quality.asr_model_path)
        self.asr_device.setCurrentText(config.quality.asr_device)
        self.asr_compute.setCurrentText(config.quality.asr_compute_type)
        for key, spin in self.qc_spins.items():
            spin.setValue(getattr(config.quality, key))
        self.trim.setChecked(config.quality.trim_silence)
        self.trim_margin.setValue(config.quality.trim_margin_ms)
        self.check_abrupt.setChecked(config.quality.check_abrupt_edges)
        self.check_repetition.setChecked(config.quality.check_repetition)
        self.keep_original.setChecked(config.keep_original)
        self.speaker_enabled.setChecked(config.quality.speaker_enabled)
        self.speaker_model.setText(config.quality.speaker_model_path)
        self.speaker_calibration.setText(config.quality.speaker_calibration_path)

    def voice_configuration(self) -> tuple[VoiceProfile, SynthesisSettings]:
        reference = Path(self.reference_path.text()).resolve()
        settings = SynthesisSettings(
            engine=self.engine.currentData(),
            device="cpu" if self.engine.currentData() == "mock" else self.device.currentText(),
            model_dir=self.model_path.text(),
            seed=self.random_seed.value(),
            service_config=self.service_config.text(),
            **{key: spin.value() for key, spin in self.synthesis_spins.items()},
        )
        voice_key = str(reference) + ":" + self.language.currentData()
        if settings.engine == "documented-service":
            voice_key += (
                ":"
                + str(Path(settings.service_config).resolve())
                + ":"
                + self.service_voice_id.text().strip()
            )
        voice = VoiceProfile(
            "voice-" + uuid.uuid5(uuid.NAMESPACE_URL, voice_key).hex,
            self.voice_name.text(),
            str(reference),
            self._reference_digest,
            self.language.currentData(),
            self.permission.isChecked(),
            model_version=settings.engine,
            synthesis_settings=asdict(settings),
            source="documented-service" if settings.engine == "documented-service" else "reference-wav",
            service_voice_id=self.service_voice_id.text().strip()
            if settings.engine == "documented-service"
            else "",
        )
        return voice, settings

    def configuration(self) -> ProjectConfig:
        if self.repo is None:
            raise ValueError("Create or open a project first")
        voice, settings = self.voice_configuration()
        config = ProjectConfig(
            self._last_config.name if self._last_config else self.repo.root.name,
            voice,
            language=voice.language,
            synthesis=settings,
            quality=QualitySettings(
                verify_asr=self.verify_asr.isChecked(),
                asr_model_path=self.asr_path.text(),
                asr_device=self.asr_device.currentText(),
                asr_compute_type=self.asr_compute.currentText(),
                **{key: spin.value() for key, spin in self.qc_spins.items()},
                trim_silence=self.trim.isChecked(),
                trim_margin_ms=self.trim_margin.value(),
                check_abrupt_edges=self.check_abrupt.isChecked(),
                check_repetition=self.check_repetition.isChecked(),
                speaker_enabled=self.speaker_enabled.isChecked(),
                speaker_model_path=self.speaker_model.text(),
                speaker_calibration_path=self.speaker_calibration.text(),
            ),
            target_count=self.target_count.value(),
            min_words=self.min_words.value(),
            max_words=self.max_words.value(),
            category_weights={key: spin.value() for key, spin in self.category_weights.items()},
            custom_topics=tuple(t.strip() for t in self.topics.text().split(",") if t.strip()),
            text_method=self.text_method.currentData(),
            llm_endpoint=self.llm_endpoint.text(),
            llm_model=self.llm_model.currentText(),
            allow_remote_llm=self.allow_remote.isChecked(),
            keep_original=self.keep_original.isChecked(),
            target_seconds=self.target_seconds.value(),
            max_retries=self.max_retries.value(),
            max_attempts=self.max_attempts.value(),
            minimum_free_gb=self.minimum_free.value(),
            minimum_gpu_free_gb=self.minimum_gpu_free.value(),
            audio_sample_rate=self.audio_rate.currentData(),
            audio_subtype=self.audio_subtype.currentText(),
            generation_workers=self.generation_workers.value(),
            auto_generation_workers=self.auto_generation_workers.isChecked(),
            overlap_quality=self.overlap_quality.isChecked(),
            require_cuda=self.require_cuda.isChecked(),
            gpu_memory_per_worker_gb=self.worker_vram.value(),
        )
        return config

    def save_project(self) -> None:
        try:
            snapshot = self.configuration()
        except Exception as exc:
            self._error(str(exc))
            return

        def task():
            # Explicit save imports/confirms a newly selected reference; later jobs use this exact hash.
            config = (
                snapshot
                if snapshot.voice.reference_sha256
                else replace(
                    snapshot,
                    voice=replace(
                        snapshot.voice, reference_sha256=sha256_file(snapshot.voice.reference_path)
                    ),
                )
            )
            config.validate()
            self.repo.save_voice(config.voice)
            self.repo.save_config(config)
            return config

        def saved(config):
            self.apply_config(config)
            self.statusBar().showMessage("Project and voice saved")
            self.saved_voices.addItem(config.voice.name, config.to_dict()["voice"])

        self.run_task(task, saved)

    def refresh_stats(self) -> None:
        if self.repo:
            stats = self.repo.statistics()
            self.stats_label.setText(
                f"Accepted {stats['accepted']}  •  Rejected {stats['rejected']}  •  Duration {stats['accepted_seconds']:.1f} s  •  Attempts {stats['attempts']}"
            )

    def play(self, path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    def run_task(self, task: Callable, handler: Callable | None = None) -> None:
        if self.runner.busy or self.aux_runner.busy or self.training_busy:
            self.statusBar().showMessage("A background operation is active; wait or stop the current job.")
            return
        self._result_handler = handler
        self._task_cancel.clear()
        self.job_selector.setEnabled(False)
        self.runner.start(task)
        self._update_voice_preview_controls()
        self._update_training_controls()

    @Slot(object)
    def _result(self, result: Any) -> None:
        if self._result_handler:
            self._result_handler(result)

    @Slot(str)
    def _error(self, message: str) -> None:
        self.batch_log.appendPlainText(message)
        self.statusBar().showMessage(message.splitlines()[-1] if message else "Operation failed")

    @Slot()
    def _finished(self) -> None:
        self.job_selector.setEnabled(True)
        if self._closing:
            self.close()

    @Slot(object)
    def _progress(self, event: dict[str, Any]) -> None:
        if runtime := event.get("runtime"):
            self._active_parallel_runtime = runtime if event.get("status") in {"RUNNING", "PAUSED"} else None
            mode = f"Auto, limit {runtime['worker_limit']}" if runtime.get("automatic_workers") else "manual"
            self.runtime_label.setText(
                f"Generation: {runtime['device']} • {runtime['workers']} worker(s) • "
                f"{mode} • {runtime.get('gpu_name', '')} • ASR: {runtime.get('asr_device', self.asr_device.currentText())}"
                + (
                    f"\nGPU tasks: {runtime['generation_active']}/{runtime['workers']} • "
                    f"Quality tasks: {runtime['quality_active']}/{runtime.get('quality_workers', 1)} • "
                    f"Queued checks: {runtime['quality_waiting']}"
                    if "generation_active" in runtime
                    else ""
                )
            )
        if "exported" in event:
            self.export_progress.setRange(0, event["total"])
            self.export_progress.setValue(event["exported"])
            return
        if event.get("audio_path"):
            self._latest_audio = event["audio_path"]
        eta = event.get("eta_seconds")
        self.throughput_label.setText(
            f"{event.get('rate', 0) * 60:.1f} accepted clips/min • ETA: {eta / 60:.1f} min"
            if eta is not None
            else f"{event.get('rate', 0) * 60:.1f} accepted clips/min • ETA: collecting data"
        )
        count = event.get("accepted", 0)
        self.progress_bar.setRange(0, self.target_count.value())
        self.progress_bar.setValue(count)
        self.current_sentence.setText(event.get("current_text", event.get("status", "")))
        if event.get("error"):
            self.batch_log.appendPlainText(str(event["error"]))
        self.stats_label.setText(
            f"Accepted {count}  •  Duration {event.get('accepted_seconds', 0):.1f} s  •  {event.get('status', '')}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.training_source_timer.stop()
        if self.runner.busy or self.aux_runner.busy or self.training_busy or self.training_inspector.busy:
            self._closing = True
            self.stop_job()
            self.training_controller.stop()
            self.statusBar().showMessage("Stopping safely; waiting for the current operation to finish…")
            event.ignore()
            return
        self.player.stop()
        self._llm_models_timer.stop()
        self._llm_models_pending = False
        self.guide_dock.hide()
        event.accept()
