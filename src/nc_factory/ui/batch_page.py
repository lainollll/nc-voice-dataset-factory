from __future__ import annotations

import json

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QInputDialog,
    QCheckBox,
)

from ..domain import ProjectConfig, TextItem
from ..pipeline import Pipeline


class BatchPageMixin:
    def _batch_page(self) -> None:
        layout = self.page(
            "Batch Generator",
            "Parallel GPU generation with bounded quality checking. Stop and pause finish in-flight clips safely.",
        )
        self.target_count = QSpinBox()
        self.target_count.setRange(0, 1_000_000)
        self.target_count.setValue(1000)
        self.job_selector = QComboBox()
        self.job_selector.currentIndexChanged.connect(self._select_job)
        layout.addWidget(QLabel("Persistent jobs (resume uses the job's saved settings)"))
        layout.addWidget(self.job_selector)
        layout.addWidget(QLabel("Target accepted recordings"))
        layout.addWidget(self.target_count)
        form = QFormLayout()
        self.target_seconds = QDoubleSpinBox()
        self.target_seconds.setRange(0, 10_000_000)
        self.max_retries, self.max_attempts = QSpinBox(), QSpinBox()
        self.max_retries.setRange(0, 20)
        self.max_retries.setValue(2)
        self.max_attempts.setRange(1, 10_000_000)
        self.max_attempts.setValue(5000)
        self.minimum_free, self.minimum_gpu_free = QDoubleSpinBox(), QDoubleSpinBox()
        self.minimum_free.setRange(0, 10000)
        self.minimum_free.setValue(2)
        self.minimum_gpu_free.setRange(0, 128)
        form.addRow("Target accepted seconds (0 disables)", self.target_seconds)
        form.addRow("Retries per source sentence", self.max_retries)
        form.addRow("Total attempt budget", self.max_attempts)
        form.addRow("Minimum free disk (GiB)", self.minimum_free)
        form.addRow("Minimum free GPU memory (GiB; 0 uses measured fit)", self.minimum_gpu_free)
        self.generation_workers = QSpinBox()
        self.generation_workers.setRange(1, 8)
        self.generation_workers.setValue(8)
        self.generation_workers.setToolTip(
            "Choose 1–8 workers. Each uses a separate GPU model instance; choose fewer if VRAM is limited."
        )
        self.overlap_quality = QCheckBox("Check previous audio while the GPU generates the next clips")
        self.overlap_quality.setChecked(True)
        self.require_cuda = QCheckBox("Require GPU for native speech generation (no CPU fallback)")
        self.require_cuda.setChecked(True)
        self.worker_vram = QDoubleSpinBox()
        self.worker_vram.setRange(0.5, 64)
        self.worker_vram.setValue(4)
        self.worker_vram.setToolTip(
            "A preflight estimate, not a memory limit. Larger models and long clips can need more. "
            "At least one GiB or the configured minimum is reserved. Auto adds four GiB if ASR runs on the GPU."
        )
        self.auto_generation_workers = QCheckBox("Automatically choose GPU workers")
        self.auto_generation_workers.setChecked(True)
        self.auto_generation_workers.setToolTip(
            "For native CUDA speech generation, choose workers when starting a new job or resuming a stopped job, "
            "using free VRAM, the per-worker estimate and reserve. The limit below is a ceiling. "
            "Each GPU worker loads its own model. Auto also selects concurrent CPU transcript checks "
            "from available CPU/RAM when speaker verification is off. Resume uses the job's saved Auto/manual mode."
        )
        self.worker_count_label = QLabel("Maximum workers in Auto")
        self.auto_generation_workers.toggled.connect(self._generation_mode_changed)
        form.addRow(self.auto_generation_workers)
        form.addRow(self.worker_count_label, self.generation_workers)
        form.addRow(self.overlap_quality)
        form.addRow(self.require_cuda)
        form.addRow("Estimated GPU GiB per worker", self.worker_vram)
        self.audio_rate = QComboBox()
        for rate in (16000, 22050, 24000, 44100, 48000):
            self.audio_rate.addItem(str(rate), rate)
        self.audio_rate.setCurrentIndex(self.audio_rate.findData(24000))
        self.audio_subtype = QComboBox()
        self.audio_subtype.addItems(["PCM_16", "PCM_24", "FLOAT"])
        form.addRow("Working WAV sample rate", self.audio_rate)
        form.addRow("Working WAV encoding (training export is PCM16)", self.audio_subtype)
        layout.addLayout(form)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setPlaceholderText("One source sentence per line")
        self.text_preview.setPlainText(
            "Please remember to bring your umbrella tomorrow.\nThe morning train arrived at the station a little early.\nCan you tell me where the nearest library is?"
        )
        layout.addWidget(self.text_preview)
        row = QHBoxLayout()
        for text, action in (
            ("Start", self.start_job),
            ("Pause", self.pause_job),
            ("Resume", self.resume_job),
            ("Stop", self.stop_job),
        ):
            row.addWidget(self.button(text, action, text == "Start"))
        layout.addLayout(row)
        self.progress_bar = QProgressBar()
        retry_row = QHBoxLayout()
        retry_row.addWidget(self.button("Retry failed / rejected samples…", self.retry_failures))
        retry_row.addWidget(self.button("Extend stopped job attempt budget…", self.extend_job_budget))
        layout.addLayout(retry_row)
        self.current_sentence = QLabel("Ready")
        self.current_sentence.setWordWrap(True)
        self.throughput_label = QLabel("Throughput / ETA require measured accepted recordings.")
        self.runtime_label = QLabel("GPU / worker status appears when a job starts.")
        self.runtime_label.setWordWrap(True)
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.document().setMaximumBlockCount(500)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.current_sentence)
        layout.addWidget(self.throughput_label)
        layout.addWidget(self.runtime_label)
        layout.addWidget(
            self.button("Play latest generated audio", lambda: self.play(getattr(self, "_latest_audio", "")))
        )
        layout.addWidget(self.batch_log)

    @Slot(bool)
    def _generation_mode_changed(self, automatic: bool) -> None:
        self.worker_count_label.setText("Maximum workers in Auto" if automatic else "Parallel generation workers")

    def start_job(self) -> None:
        if self.runner.busy or self.aux_runner.busy or self.training_busy:
            self.statusBar().showMessage(
                "An operation is active; the running pipeline and its controls are unchanged."
            )
            return
        try:
            config = self.configuration()
            texts = self.text_preview.toPlainText().splitlines()
            preview_items = tuple(self._preview_items)
            if any(item.language != config.language for item in preview_items):
                raise ValueError(
                    "Preview language differs from the selected voice language. Generate/import the text again."
                )
            if not any(t.strip() for t in texts):
                raise ValueError("Add source sentences first")
        except Exception as exc:
            self._error(str(exc))
            return
        pipeline = Pipeline(self.repo, on_progress=self.progress_event.emit)
        self.pipeline = pipeline
        repo = self.repo

        def task():
            from ..text import ProjectTextSource, normalize_text

            existing = {t.spoken_text: t for t in preview_items}
            items = [
                existing[t.strip()]
                if t.strip() in existing and existing[t.strip()].language == config.language
                else TextItem(
                    t.strip(),
                    normalize_text(t.strip(), config.language),
                    language=config.language,
                    provenance={"method": "manual"},
                )
                for t in texts
                if t.strip()
            ]
            items.extend(preview_items[500:])
            repo.save_config(config)
            job_id = repo.create_job(config, items)
            pipeline.text_source = ProjectTextSource(config, repo, job_id)
            return {**pipeline.run(job_id), "job_id": job_id}

        self.run_task(task, self._job_finished)

    def pause_job(self) -> None:
        if self.pipeline:
            self.pipeline.request_pause()

    def resume_job(self) -> None:
        if self.training_busy:
            self.statusBar().showMessage("Stop fine-tuning before resuming speech generation.")
            return
        if self.pipeline and self.runner.busy:
            self.pipeline.request_resume()
        elif self.repo and self.job_id:
            self.pipeline = Pipeline(self.repo, on_progress=self.progress_event.emit)
            job_id = self.job_id

            def task():
                from ..text import ProjectTextSource

                cfg = ProjectConfig.from_dict(json.loads(self.repo.get_job(job_id)["config_json"]))
                self.pipeline.text_source = ProjectTextSource(cfg, self.repo, job_id)
                return self.pipeline.run(job_id)

            self.run_task(task, self._job_finished)

    def _select_job(self) -> None:
        if not self.runner.busy:
            self.job_id = self.job_selector.currentData()

    def retry_failures(self) -> None:
        if not self.repo or not self.job_id or self.runner.busy:
            return
        reason, ok = QInputDialog.getText(
            self, "Retry failures", "Reason for retrying this job's failed/rejected samples:"
        )
        if not ok or not reason.strip():
            return
        repo, job_id = self.repo, self.job_id

        def task():
            count = 0
            with repo.writer_lock():
                for sample in list(repo.iter_samples(job_id=job_id)):
                    if sample["state"] in {"REJECTED", "FAILED"}:
                        repo.review(sample["id"], "retry", reason)
                        count += 1
            return count

        self.run_task(
            task,
            lambda count: self.statusBar().showMessage(
                f"Queued {count} samples. Resume the selected job; its saved model/voice and remaining budget apply."
            ),
        )

    def extend_job_budget(self) -> None:
        if not self.repo or not self.job_id or self.runner.busy:
            return
        additional, ok = QInputDialog.getInt(
            self,
            "Extend attempt budget",
            "Additional attempts (does not change voice/model/quality settings):",
            1000,
            1,
            10_000_000,
        )
        if not ok:
            return
        reason, ok = QInputDialog.getText(
            self, "Budget extension reason", "Reason (stored in project audit history):"
        )
        if ok and reason.strip():
            repo, job_id = self.repo, self.job_id
            self.run_task(
                lambda: repo.extend_attempt_budget(job_id, additional, reason),
                lambda _: self.statusBar().showMessage(
                    "Attempt budget extended. Resume the selected job explicitly."
                ),
            )

    def stop_job(self) -> None:
        self._task_cancel.set()
        if self.pipeline:
            self.pipeline.request_stop()

    def _job_finished(self, result: dict) -> None:
        if result.get("job_id"):
            self.job_id = result["job_id"]
        self.batch_log.appendPlainText(json.dumps(result, ensure_ascii=False))
        self.refresh_stats()
        self.refresh_browser()
        if self.repo:
            current = self.job_id
            self.job_selector.blockSignals(True)
            self.job_selector.clear()
            for job in self.repo.list_jobs():
                self.job_selector.addItem(
                    f"{job['created_at'][:19]} • {job['status']} • {job['id'][:8]}", job["id"]
                )
            self.job_selector.setCurrentIndex(self.job_selector.findData(current))
            self.job_selector.blockSignals(False)
