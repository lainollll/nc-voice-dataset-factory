from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
)

from ..export import ExportOptions, export_dataset


class ExportPageMixin:
    def _export_page(self) -> None:
        layout = self.page(
            "Export", "Export an immutable snapshot of accepted recordings with exact text/audio pairing."
        )
        self.export_formats = {}
        for key, label in (
            ("csv", "Standard CSV"),
            ("ljspeech", "LJSpeech metadata"),
            ("lfm3", "LFM3 notebook loader + JSONL"),
            ("huggingface", "Hugging Face DatasetDict"),
            ("ljspeech_parquet", "LJ Speech-style Hugging Face Parquet (full split, 22,050 Hz)"),
        ):
            check = QCheckBox(label)
            if key == "ljspeech_parquet":
                check.setToolTip(
                    "Matches MikhailT/lj-speech's four-column Parquet layout. Creates ljspeech_hf/data with embedded 22,050 Hz audio and a full split."
                )
            check.setChecked(key in ExportOptions().formats)
            self.export_formats[key] = check
            layout.addWidget(check)
        form = QFormLayout()
        self.validation_fraction = QDoubleSpinBox()
        self.validation_fraction.setRange(0, 0.5)
        self.validation_fraction.setSingleStep(0.05)
        self.validation_fraction.setValue(0.1)
        self.split_seed = QSpinBox()
        self.split_seed.setRange(0, 2**31 - 1)
        self.export_name = QLineEdit("NC_Voice_Dataset")
        self.export_rejected = QCheckBox(
            "Explicitly include rejected audio in a separate rejected folder (never training splits)"
        )
        form.addRow("New export folder name", self.export_name)
        form.addRow("Validation fraction", self.validation_fraction)
        form.addRow("Deterministic split seed", self.split_seed)
        layout.addLayout(form)
        layout.addWidget(self.export_rejected)
        self.export_progress = QProgressBar()
        layout.addWidget(self.export_progress)
        layout.addWidget(self.button("Export accepted dataset…", self.export_project, True))
        layout.addWidget(self.button("Validate an existing export…", self.validate_dataset))
        layout.addWidget(self.button("Open LFM3 fine-tuning", lambda: self.open_page("LFM3 Fine-tuning")))
        self.export_status = QLabel("Choose a new destination folder for each export.")
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)
        layout.addStretch()

    def validate_dataset(self) -> None:
        from ..export import validate_export

        path = QFileDialog.getExistingDirectory(self, "Validate exported dataset")
        if path:
            self.run_task(
                lambda: validate_export(Path(path)),
                lambda result: self.export_status.setText(json.dumps(result, ensure_ascii=False, indent=2)),
            )

    def export_project(self) -> None:
        if not self.repo:
            return
        parent = QFileDialog.getExistingDirectory(self, "Parent folder for a new export")
        if parent:
            name = self.export_name.text().strip()
            if not name or Path(name).name != name or name in {".", ".."}:
                self._error("Enter a single folder name for the new export")
                return
            destination = Path(parent) / name
            options = ExportOptions(
                tuple(key for key, check in self.export_formats.items() if check.isChecked()),
                self.validation_fraction.value(),
                self.split_seed.value(),
                self.export_rejected.isChecked(),
            )
            repo = self.repo
            self.run_task(
                lambda: export_dataset(repo, destination, options, self.progress_event.emit),
                self._export_ready,
            )

    def _export_ready(self, report) -> None:
        self.export_status.setText(f"Exported {report.accepted_count} recordings to {report.destination}")
        self.training_dataset.setText(report.destination)
