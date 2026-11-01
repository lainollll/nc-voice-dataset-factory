from __future__ import annotations

import uuid
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
)

from ..audio import write_wav_atomic
from ..domain import SynthesisSettings
from ..tts import create_tts


class VoicePageMixin:
    def _voice_page(self) -> None:
        layout = self.page(
            "Voice Manager",
            "Use a permitted reference recording; no separately trained voice checkpoint is required.",
        )
        form = QFormLayout()
        self.voice_name = QLineEdit("My voice")
        self.saved_voices = QComboBox()
        self.saved_voices.addItem("Select a saved project voice…", None)
        self.saved_voices.currentIndexChanged.connect(self._load_saved_voice)
        self.reference_path = QLineEdit()
        self.reference_path.textChanged.connect(lambda: setattr(self, "_reference_digest", ""))
        self.model_path = QLineEdit()
        self.language = QComboBox()
        self.language.addItem("English", "en")
        self.language.addItem("Swedish", "sv")
        self.engine = QComboBox()
        for label, value in (
            ("Chatterbox Multilingual V3", "multilingual-v3"),
            ("Multilingual V2", "multilingual-v2"),
            ("Chatterbox English", "english"),
            ("Documented local HTTP service", "documented-service"),
            ("TEST ONLY — synthetic tones", "mock"),
        ):
            self.engine.addItem(label, value)
        form.addRow("Voice name", self.voice_name)
        form.addRow("Saved voice profiles", self.saved_voices)
        form.addRow("Reference WAV", self.reference_path)
        form.addRow("", self.button("Import reference WAV…", self.choose_reference))
        form.addRow("Language", self.language)
        form.addRow("Model", self.engine)
        form.addRow("Local model folder", self.model_path)
        form.addRow("", self.button("Choose model folder…", self.choose_model))
        self.service_config = QLineEdit()
        self.service_config.textChanged.connect(self._clear_service_profiles)
        form.addRow("Documented service mapping JSON", self.service_config)
        form.addRow("", self.button("Choose service contract mapping…", self.choose_service))
        self.service_voice_id = QLineEdit()
        self.service_voice_id.setPlaceholderText("Explicit provider voice ID, if the API uses one")
        self.service_voices = QComboBox()
        self.service_voices.addItem("Select a documented service voice…", None)
        self.service_voices.currentIndexChanged.connect(self._apply_service_voice)
        form.addRow("Service voice ID", self.service_voice_id)
        form.addRow("Connected voice profiles", self.service_voices)
        form.addRow("", self.button("Fetch documented service voices", self.fetch_service_profiles))
        self.permission = QCheckBox("I own or have permission to use this voice")
        form.addRow(self.permission)
        self.device = QComboBox()
        self.device.addItems(["cuda", "cpu"])
        form.addRow("Synthesis device", self.device)
        self.synthesis_spins = {}
        for key, label, value, low, high in (
            ("temperature", "Temperature", 0.8, 0.01, 2),
            ("exaggeration", "Exaggeration", 0.5, 0, 2),
            ("cfg_weight", "CFG weight", 0.5, 0, 2),
            ("repetition_penalty", "Repetition penalty", 1.2, 1, 3),
            ("top_p", "Top-p", 1, 0.01, 1),
            ("min_p", "Min-p", 0.05, 0, 1),
            ("variation", "Temperature variation (max ±0.1)", 0, 0, 0.1),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(low, high)
            spin.setSingleStep(0.01)
            spin.setValue(value)
            self.synthesis_spins[key] = spin
            form.addRow(label, spin)
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 2**31 - 1)
        form.addRow("Random seed", self.random_seed)
        layout.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self.button("Save voice / project", self.save_project))
        row.addWidget(self.button("Test Voice", self.test_voice, True))
        row.addWidget(self.button("Play reference", lambda: self.play(self.reference_path.text())))
        layout.addLayout(row)
        download_row = QHBoxLayout()
        self.download_kind = QComboBox()
        self.download_kind.addItems(["multilingual-v3", "multilingual-v2", "english", "asr", "speaker"])
        download_row.addWidget(self.download_kind)
        download_row.addWidget(self.button("Inspect model download…", self.inspect_model_download))
        download_row.addWidget(self.button("Download inspected model…", self.download_inspected_model))
        layout.addLayout(download_row)
        self.download_status = QLabel("Model weights are never downloaded automatically.")
        self.download_status.setWordWrap(True)
        layout.addWidget(self.download_status)
        self._download_plan = None
        layout.addStretch()

    def inspect_model_download(self) -> None:
        from ..models import model_download_plan

        kind = self.download_kind.currentText()
        self.statusBar().showMessage("Inspecting the explicitly requested model repository…")

        def ready(plan):
            self._download_plan = plan
            self.download_status.setText(
                f"{plan['repository']}\nRevision: {plan['revision']} • License: {plan['license']} • Size: {plan['bytes'] / 1024**3:.2f} GiB\n{plan['source']}\nClick Download inspected model to fetch these files into a chosen empty folder."
            )

        self.run_task(lambda: model_download_plan(kind), ready)

    def download_inspected_model(self) -> None:
        from ..models import download_model

        if not self._download_plan:
            self._error("Inspect a model first so its source, license, size and revision are visible")
            return
        path = QFileDialog.getExistingDirectory(self, "Choose an empty destination for model weights")
        if path:
            plan = dict(self._download_plan)

            def ready(value):
                if plan["kind"] == "asr":
                    self.asr_path.setText(value)
                elif plan["kind"] == "speaker":
                    self.speaker_model.setText(value)
                else:
                    self.model_path.setText(value)
                    self.engine.setCurrentIndex(self.engine.findData(plan["kind"]))
                self.download_status.setText(
                    f"Downloaded {plan['repository']} at {plan['revision']} to {value}"
                )

            self.run_task(lambda: download_model(plan, Path(path)), ready)

    def _load_saved_voice(self) -> None:
        value = self.saved_voices.currentData()
        if not value:
            return
        self.voice_name.setText(value["name"])
        self.reference_path.setText(value["reference_path"])
        self._reference_digest = value["reference_sha256"]
        self.language.setCurrentIndex(self.language.findData(value["language"]))
        self.permission.setChecked(value["permission_confirmed"])
        settings = value.get("synthesis_settings")
        if settings:
            self.engine.setCurrentIndex(self.engine.findData(settings["engine"]))
            self.model_path.setText(settings.get("model_dir", ""))
            self.service_config.setText(settings.get("service_config", ""))
            self.device.setCurrentText(settings.get("device", "cuda"))
            self.random_seed.setValue(settings.get("seed", 0))
            for key, spin in self.synthesis_spins.items():
                if key in settings:
                    spin.setValue(settings[key])
        self.service_voice_id.setText(value.get("service_voice_id", ""))

    def _clear_service_profiles(self) -> None:
        self.service_voice_id.clear()
        self.service_voices.blockSignals(True)
        self.service_voices.clear()
        self.service_voices.addItem("Select a documented service voice…", None)
        self.service_voices.blockSignals(False)

    def fetch_service_profiles(self) -> None:
        if self.engine.currentData() != "documented-service":
            self._error("Select the documented local HTTP service engine first")
            return
        settings = SynthesisSettings(engine="documented-service", service_config=self.service_config.text())

        def task():
            backend = create_tts(settings)
            try:
                return backend.mapped_profiles()
            finally:
                backend.close()

        def ready(profiles):
            if (
                self.service_config.text() != settings.service_config
                or self.engine.currentData() != "documented-service"
            ):
                self.statusBar().showMessage("Service selection changed; fetched profiles were discarded")
                return
            self._clear_service_profiles()
            for profile in profiles:
                self.service_voices.addItem(f"{profile['name']} · {profile['service_voice_id']}", profile)
            self.statusBar().showMessage(f"Fetched {len(profiles)} documented service voices")

        self.run_task(task, ready)

    def _apply_service_voice(self) -> None:
        profile = self.service_voices.currentData()
        if not profile:
            return
        self.voice_name.setText(profile["name"])
        if profile.get("reference_path"):
            self.reference_path.setText(profile["reference_path"])
        if profile.get("language"):
            self.language.setCurrentIndex(self.language.findData(profile["language"]))
        self.service_voice_id.setText(profile["service_voice_id"])
        self.permission.setChecked(False)

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Reference WAV", filter="WAV audio (*.wav)")
        if path:
            self.reference_path.setText(path)
            self.permission.setChecked(False)

    def choose_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Local Chatterbox weights")
        if path:
            self.model_path.setText(path)

    def choose_service(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Documented service mapping", filter="JSON (*.json)")
        if path:
            self.service_config.setText(path)

    def test_voice(self) -> None:
        try:
            config = self.configuration()
        except Exception as exc:
            self._error(str(exc))
            return

        def task():
            config.validate()
            backend = create_tts(config.synthesis)
            try:
                text = (
                    "Hej, det här är ett kort röstprov."
                    if config.language == "sv"
                    else "Hello, this is a short voice test."
                )
                result = backend.synthesize(text, config.voice, config.synthesis, config.synthesis.seed)
                path = self.repo.root / "previews" / f"{uuid.uuid4().hex}.wav"
                write_wav_atomic(path, result.samples, result.sample_rate)
                return str(path)
            finally:
                backend.close()

        self.run_task(task, self.play)
