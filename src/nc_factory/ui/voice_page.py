from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, Slot
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
    QGroupBox,
    QPlainTextEdit,
    QVBoxLayout,
    QSlider,
    QWidget,
)

from ..audio import sha256_file, write_wav_atomic
from ..domain import SynthesisSettings
from ..quick_setup import APPLICATION_ROOT, LocalModels, discover_models
from ..tts import create_tts
from .sections import CollapsibleSection

PREVIEW_TEXT = {"en": "Hello, this is a short voice test.", "sv": "Hej, det här är ett kort röstprov."}


class VoicePageMixin:
    def _voice_page(self) -> None:
        layout = self.page(
            "Voice Manager",
            "Choose a reference voice, adjust Chatterbox settings, and listen before creating a dataset.",
        )
        reference = QGroupBox("1. Choose your reference voice")
        reference_layout = QVBoxLayout(reference)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.voice_name = QLineEdit("My voice")
        self.saved_voices = QComboBox()
        self.saved_voices.addItem("Select a saved project voice…", None)
        self.saved_voices.currentIndexChanged.connect(self._load_saved_voice)
        self.reference_path = QLineEdit()
        self.reference_path.textChanged.connect(self._voice_reference_changed)
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
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_path, 1)
        reference_row.addWidget(self.button("Choose WAV…", self.choose_reference))
        form.addRow("Reference WAV", reference_row)
        form.addRow("Language", self.language)
        form.addRow("Model", self.engine)
        self.engine.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.engine.setMinimumContentsLength(10)
        form.addRow("", self.button("Find local model", self.find_preview_models))
        self.permission = QCheckBox("I have permission to use this voice")
        self.permission.setToolTip("Confirm that you own this voice or have the owner's permission to use it.")
        form.addRow(self.permission)
        reference_layout.addLayout(form)
        self.voice_profile_details = CollapsibleSection("Saved profiles and model folders")
        profile_form = QFormLayout()
        profile_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.saved_voices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.saved_voices.setMinimumContentsLength(10)
        profile_form.addRow("Saved voice profiles", self.saved_voices)
        profile_form.addRow("Local model folder", self.model_path)
        profile_form.addRow("", self.button("Choose model folder…", self.choose_model))
        self.voice_profile_details.body.addLayout(profile_form)
        reference_layout.addWidget(self.voice_profile_details)
        layout.addWidget(reference)

        preview = QGroupBox("2. Adjust and preview")
        preview_layout = QVBoxLayout(preview)
        common = QFormLayout()
        common.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.synthesis_spins = {}
        self.synthesis_sliders: dict[str, QSlider] = {}
        self.voice_advanced = CollapsibleSection("Advanced speech settings")
        advanced = QFormLayout()
        advanced.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, value, low, high, hint in (
            ("temperature", "Variation in delivery", 0.8, 0.01, 2, "Chatterbox temperature. Higher values allow more variation; compare using the same text and seed."),
            ("exaggeration", "Expressiveness", 0.5, 0, 2, "Chatterbox exaggeration. Controls expressive delivery; results depend on the reference and model."),
            ("cfg_weight", "Voice guidance", 0.5, 0, 2, "Chatterbox CFG weight. Compare different values while keeping the other settings fixed."),
            ("repetition_penalty", "Repetition penalty", 1.2, 1, 3, "Discourages repeated generated tokens."),
            ("top_p", "Top-p", 1, 0.01, 1, "Sampling probability cutoff."),
            ("min_p", "Min-p", 0.05, 0, 1, "Minimum sampling probability."),
            ("variation", "Dataset temperature variation", 0, 0, 0.1, "Per-attempt temperature variation for dataset generation. Single previews use the temperature above."),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(low, high)
            spin.setSingleStep(0.01)
            spin.setValue(value)
            spin.setToolTip(hint)
            spin.setAccessibleName(label)
            self.synthesis_spins[key] = spin
            if key in {"temperature", "exaggeration", "cfg_weight"}:
                control = QWidget()
                control.setObjectName("speechControlRow")
                controls = QHBoxLayout(control)
                controls.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(round(low * 1000), round(high * 1000))
                slider.setMinimumWidth(90)
                slider.setSingleStep(10)
                slider.setPageStep(100)
                slider.setValue(round(value * 1000))
                slider.setAccessibleName(label)
                slider.setToolTip(hint)
                slider.valueChanged.connect(lambda number, target=spin: target.setValue(number / 1000))
                spin.valueChanged.connect(lambda number, target=slider: target.setValue(round(number * 1000)))
                self.synthesis_sliders[key] = slider
                spin.setMaximumWidth(100)
                controls.addWidget(slider, 1)
                controls.addWidget(spin)
                common.addRow(label, control)
            else:
                advanced.addRow(label, spin)
        preview_layout.addLayout(common)
        self.preview_text = QPlainTextEdit(PREVIEW_TEXT["en"])
        self.preview_text.setAccessibleName("Voice preview text")
        self.preview_text.setMinimumHeight(70)
        self.preview_text.setMaximumHeight(85)
        preview_layout.addWidget(QLabel("Words to preview"))
        preview_layout.addWidget(self.preview_text)
        self.preview_button = self.button("Generate voice preview", self.test_voice, True)
        preview_layout.addWidget(self.preview_button)
        playback = QHBoxLayout()
        self.preview_replay = self.button("Replay", self._replay_voice_preview)
        self.preview_replay.setAccessibleName("Replay last generated preview")
        playback.addWidget(self.preview_replay)
        stop_audio = self.button("Stop", self.player.stop)
        stop_audio.setAccessibleName("Stop audio playback")
        playback.addWidget(stop_audio)
        reference_audio = self.button("Reference", lambda: self.play(self.reference_path.text()))
        reference_audio.setAccessibleName("Play the original reference recording")
        playback.addWidget(reference_audio)
        preview_layout.addLayout(playback)
        self.preview_status = QLabel("Choose a reference WAV and confirm voice permission to preview it.")
        self.preview_status.setWordWrap(True)
        self.preview_status.setTextFormat(Qt.TextFormat.PlainText)
        preview_layout.addWidget(self.preview_status)
        note = QLabel("This previews the reference voice, not a trained LFM3 adapter, and adds no dataset records. "
                      "Save voice / project keeps these settings for new Batch Generator jobs. "
                      "Getting Started uses its automatic dataset settings.")
        note.setWordWrap(True)
        preview_layout.addWidget(note)
        layout.addWidget(preview)

        self.device = QComboBox()
        self.device.addItems(["cuda", "cpu"])
        advanced.addRow("Synthesis device", self.device)
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 2**31 - 1)
        self.random_seed.setToolTip("Keep the same seed when comparing settings.")
        advanced.addRow("Random seed", self.random_seed)
        self.voice_advanced.body.addLayout(advanced)
        layout.addWidget(self.voice_advanced)
        layout.addWidget(self.button("Save voice / project", self.save_project))

        self.voice_service = CollapsibleSection("Documented service settings")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        self.voice_service.body.addLayout(form)
        self.voice_service.hide()
        self.engine.currentIndexChanged.connect(self._voice_engine_changed)
        layout.addWidget(self.voice_service)
        downloads = self.voice_downloads = CollapsibleSection("Model downloads")
        self.download_kind = QComboBox()
        self.download_kind.addItems(["multilingual-v3", "multilingual-v2", "english", "asr", "speaker"])
        downloads.body.addWidget(self.download_kind)
        downloads.body.addWidget(self.button("Inspect model download…", self.inspect_model_download))
        downloads.body.addWidget(self.button("Download inspected model…", self.download_inspected_model))
        self.download_status = QLabel("Model weights are never downloaded automatically.")
        self.download_status.setWordWrap(True)
        downloads.body.addWidget(self.download_status)
        layout.addWidget(downloads)
        self._download_plan = None
        self._voice_preview_active = False
        self._last_voice_preview = ""
        self._voice_preview_reference = ""
        self.language.currentIndexChanged.connect(self._preview_language_changed)
        self.reference_path.textChanged.connect(self._update_voice_preview_controls)
        self.permission.toggled.connect(self._update_voice_preview_controls)
        self.model_path.textChanged.connect(self._update_voice_preview_controls)
        self.engine.currentIndexChanged.connect(self._update_voice_preview_controls)
        self.runner.error.connect(self._voice_preview_error)
        self.runner.finished.connect(self._voice_preview_finished)
        self.aux_runner.finished.connect(self._update_voice_preview_controls)
        self._update_voice_preview_controls()
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
            self._reference_digest = ""
            self.permission.setChecked(False)

    def choose_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Local Chatterbox weights")
        if path:
            self.model_path.setText(path)

    def choose_service(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Documented service mapping", filter="JSON (*.json)")
        if path:
            self.service_config.setText(path)

    def find_preview_models(self) -> None:
        if self.runner.busy or self.aux_runner.busy or self.training_busy or self._closing:
            self.preview_status.setText("Wait for the current operation before finding a model.")
            return
        language, model, engine = self.language.currentData(), self.model_path.text(), self.engine.currentData()
        preferred = (Path(model),) if model.strip() else ()

        def ready(found: LocalModels) -> None:
            if (self.language.currentData(), self.model_path.text(), self.engine.currentData()) != (language, model, engine):
                self.preview_status.setText("The model selection changed. Find local model again to refresh.")
                return
            if found.tts_path is None:
                self.model_path.clear()
                self.voice_downloads.setExpanded(True)
                self.download_kind.setCurrentText(found.engine)
                self.preview_status.setText("No matching local speech model found. Inspect and download a model below.")
            else:
                self.engine.setCurrentIndex(self.engine.findData(found.engine))
                self.model_path.setText(str(found.tts_path))
                self.preview_status.setText("Local Chatterbox model found. Choose a WAV, confirm permission, and generate a preview.")

        self.preview_status.setText("Looking for existing local models… No model weights will be downloaded.")
        self.run_task(lambda: discover_models(APPLICATION_ROOT, language, preferred=preferred), ready)

    def test_voice(self) -> None:
        if self.runner.busy or self.aux_runner.busy or self.training_busy or self._closing:
            self.preview_status.setText("Wait for the current operation or training to finish before generating a preview.")
            return
        try:
            voice, settings = self.voice_configuration()
            text = self.preview_text.toPlainText().strip()
            if not text or len(text) > 2000:
                raise ValueError("Enter 1–2,000 characters of preview text.")
            if not voice.permission_confirmed:
                raise ValueError("Confirm that you own or have permission to use this reference voice.")
            if settings.engine == "english" and voice.language != "en":
                raise ValueError("Swedish requires a multilingual Chatterbox model.")
        except Exception as exc:
            self.preview_status.setText(str(exc))
            return
        # Capture everything before dispatch. Editing controls or changing projects
        # must not redirect a running preview or alter its parameters.
        output_root = self.repo.root if self.repo is not None else APPLICATION_ROOT

        def task():
            preview_voice = voice if voice.reference_sha256 else replace(
                voice, reference_sha256=sha256_file(voice.reference_path)
            )
            preview_voice.validate()
            settings.validate()
            backend = create_tts(settings)
            try:
                result = backend.synthesize(text, preview_voice, settings, settings.seed)
                path = output_root / "previews" / f"{uuid.uuid4().hex}.wav"
                write_wav_atomic(path, result.samples, result.sample_rate)
                return str(path)
            finally:
                backend.close()

        self._voice_preview_active = True
        self._voice_preview_reference = voice.reference_path
        self.preview_status.setText("Generating a voice preview… The first preview loads the selected model.")
        self.run_task(task, self._voice_preview_ready)
        self._update_voice_preview_controls()

    @Slot(str)
    def _voice_preview_ready(self, path: str) -> None:
        self._last_voice_preview = path
        self.preview_status.setText(
            f"Preview ready for {Path(self._voice_preview_reference).name}. "
            "Replay it or adjust settings and generate another. Not added to the dataset."
        )
        self.play(path)

    @Slot()
    def _replay_voice_preview(self) -> None:
        if self._last_voice_preview:
            self.play(self._last_voice_preview)

    @Slot(str)
    def _voice_preview_error(self, message: str) -> None:
        if self._voice_preview_active:
            self.preview_status.setText(message.strip().splitlines()[-1])

    @Slot()
    def _voice_preview_finished(self) -> None:
        self._voice_preview_active = False
        self._update_voice_preview_controls()

    def _update_voice_preview_controls(self, *_args: object) -> None:
        busy = self.runner.busy or self.aux_runner.busy or self.training_busy or self._closing
        needs_model = self.engine.currentData() in {"english", "multilingual-v2", "multilingual-v3"} and not self.model_path.text().strip()
        self.preview_button.setEnabled(not busy and not needs_model and bool(self.reference_path.text().strip()) and self.permission.isChecked())
        self.preview_replay.setEnabled(bool(self._last_voice_preview) and not self._voice_preview_active)
        if _args and not self._voice_preview_active and self.reference_path.text().strip() and self.permission.isChecked() and needs_model:
            self.preview_status.setText("Find a local Chatterbox model or choose its folder before generating a preview.")

    @Slot(str)
    def _voice_reference_changed(self, _text: str) -> None:
        self._reference_digest = ""
        self.permission.setChecked(False)
        self._last_voice_preview = ""
        if hasattr(self, "preview_status") and not self._voice_preview_active:
            self.preview_status.setText("Reference changed. Confirm permission for this voice before generating a preview.")

    def _preview_language_changed(self, *_args: object) -> None:
        if self.preview_text.toPlainText().strip() in {"", *PREVIEW_TEXT.values()}:
            self.preview_text.setPlainText(PREVIEW_TEXT[self.language.currentData()])

    def _voice_engine_changed(self, *_args: object) -> None:
        service = self.engine.currentData() == "documented-service"
        self.voice_service.setVisible(service)
        self.voice_service.setExpanded(service)
