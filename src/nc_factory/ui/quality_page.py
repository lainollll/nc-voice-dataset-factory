from __future__ import annotations

import uuid
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
)


class QualityPageMixin:
    def _quality_page(self) -> None:
        layout = self.page(
            "Quality Control",
            "Every enabled check must pass before automatic acceptance. ASR never replaces the source transcript.",
        )
        form = QFormLayout()
        self.verify_asr = QCheckBox("Verify transcripts with faster-whisper")
        self.verify_asr.setChecked(True)
        self.asr_path = QLineEdit()
        self.asr_device = QComboBox()
        self.asr_device.addItems(["cpu", "cuda"])
        self.asr_compute = QComboBox()
        self.asr_compute.addItems(["int8", "float16", "int8_float16", "float32"])
        form.addRow(self.verify_asr)
        form.addRow("Local ASR model folder", self.asr_path)
        form.addRow("", self.button("Choose ASR model…", lambda: self._choose_directory(self.asr_path)))
        form.addRow("ASR device", self.asr_device)
        form.addRow("Compute type", self.asr_compute)
        self.qc_spins = {}
        for key, label, value, maximum in (
            ("max_wer", "Maximum WER", 0.10, 1),
            ("max_cer", "Maximum CER", 0.10, 1),
            ("min_duration", "Minimum duration (s)", 3, 120),
            ("max_duration", "Maximum duration (s)", 15, 120),
            ("max_silence_ratio", "Maximum silence ratio", 0.5, 1),
            ("min_rms", "Minimum signal RMS", 0.003, 1),
            ("max_clipping_ratio", "Maximum clipping ratio", 0.001, 1),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(4 if maximum == 1 else 2)
            spin.setRange(0, maximum)
            spin.setSingleStep(0.01 if maximum == 1 else 1)
            spin.setValue(value)
            self.qc_spins[key] = spin
            form.addRow(label, spin)
        self.trim = QCheckBox("Conservative silence trim (120 ms margins)")
        self.trim.setText("Conservative silence trim")
        self.trim_margin = QSpinBox()
        self.trim_margin.setRange(0, 2000)
        self.trim_margin.setValue(120)
        self.check_abrupt = QCheckBox("Check abrupt cutoff heuristic")
        self.check_abrupt.setChecked(True)
        self.check_repetition = QCheckBox("Check repeated-audio heuristic")
        self.check_repetition.setChecked(True)
        self.speaker_enabled = QCheckBox("Require independently calibrated speaker verification")
        self.speaker_model = QLineEdit()
        self.speaker_calibration = QLineEdit()
        self.keep_original = QCheckBox("Retain original generated audio")
        self.keep_original.setChecked(True)
        for check in (self.trim, self.keep_original, self.speaker_enabled):
            form.addRow(check)
        form.addRow("Trim margin (ms)", self.trim_margin)
        form.addRow(self.check_abrupt)
        form.addRow(self.check_repetition)
        form.addRow("Local SpeechBrain speaker model", self.speaker_model)
        form.addRow(
            "", self.button("Choose speaker model…", lambda: self._choose_directory(self.speaker_model))
        )
        form.addRow("Calibration JSON", self.speaker_calibration)
        form.addRow("", self.button("Choose calibration JSON…", self._choose_calibration))
        layout.addLayout(form)
        layout.addWidget(self.button("Calibrate from labeled recordings…", self.calibrate_speaker))
        layout.addWidget(self.button("Save quality settings", self.save_project, True))
        layout.addStretch()

    def _choose_directory(self, field: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose local model folder")
        if path:
            field.setText(path)

    def _choose_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Speaker calibration", filter="JSON (*.json)")
        if path:
            self.speaker_calibration.setText(path)

    def calibrate_speaker(self) -> None:
        try:
            cfg = self.configuration()
        except Exception as exc:
            self._error(str(exc))
            return
        positives, _ = QFileDialog.getOpenFileNames(
            self, "Same-speaker recordings (distinct from reference)", filter="WAV (*.wav)"
        )
        if not positives:
            return
        negatives, _ = QFileDialog.getOpenFileNames(
            self, "Other-speaker recordings (permission required)", filter="WAV (*.wav)"
        )
        if not negatives:
            return

        def task():
            from ..speaker import calibrate_reference
            from ..storage import atomic_json

            calibration = calibrate_reference(
                cfg.quality.speaker_model_path,
                cfg.voice,
                [Path(p) for p in positives],
                [Path(p) for p in negatives],
            )
            destination = self.repo.root / "calibration" / f"{uuid.uuid4().hex}.json"
            atomic_json(destination, calibration)
            return str(destination)

        self.run_task(task, self.speaker_calibration.setText)
