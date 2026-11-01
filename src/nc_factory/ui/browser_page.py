from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QInputDialog,
    QGridLayout,
)

from .waveform import WaveformWidget, prepare_waveform


class BrowserPageMixin:
    def _browser_page(self) -> None:
        layout = self.page(
            "Dataset Browser",
            "Listen, inspect, search and review each recording. Transcript edits require revalidation.",
        )
        row = QHBoxLayout()
        self.browser_search = QLineEdit()
        self.browser_search.setPlaceholderText("Search transcript, category or ASR output")
        self.browser_filter = QComboBox()
        self.browser_filter.addItem("All states", None)
        for state in ("ACCEPTED", "REJECTED", "FAILED", "PENDING"):
            self.browser_filter.addItem(state, state)
        row.addWidget(self.browser_search, 1)
        row.addWidget(self.browser_filter)
        row.addWidget(self.button("Search / Refresh", self.refresh_browser))
        row.addWidget(self.button("Random accepted review", self.random_review))
        layout.addLayout(row)
        self.browser_table = QTableWidget(0, 9)
        self.browser_table.setHorizontalHeaderLabels(
            [
                "Filename",
                "Training text",
                "Duration",
                "Language",
                "Category",
                "Status",
                "WER",
                "CER",
                "Speaker",
            ]
        )
        self.browser_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.browser_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.browser_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.browser_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.browser_table.itemSelectionChanged.connect(self._browser_selected)
        layout.addWidget(self.browser_table, 2)
        page_row = QHBoxLayout()
        page_row.addWidget(self.button("Previous 200", lambda: self.browser_page(-1)))
        self.browser_page_label = QLabel("Page 1")
        page_row.addWidget(self.browser_page_label)
        page_row.addWidget(self.button("Next 200", lambda: self.browser_page(1)))
        layout.addLayout(page_row)
        self.waveform = WaveformWidget()
        self.waveform.setMinimumHeight(90)
        layout.addWidget(self.waveform)
        self.browser_transcript = QPlainTextEdit()
        self.browser_transcript.setMaximumHeight(95)
        self.browser_details = QPlainTextEdit()
        self.browser_details.setReadOnly(True)
        self.browser_details.setMaximumHeight(140)
        layout.addWidget(self.browser_transcript)
        layout.addWidget(self.browser_details)
        actions = QGridLayout()
        for index, (label, callback) in enumerate(
            (
                ("Play", self.play_selected),
                ("Stop audio", self.player.stop),
                ("Save transcript", self.edit_selected),
                ("Revalidate", self.revalidate_selected),
                ("Accept", lambda: self.review_selected("accept")),
                ("Reject", lambda: self.review_selected("reject")),
                ("Retry", lambda: self.review_selected("retry")),
                ("Regenerate edited text", lambda: self.review_selected("regenerate")),
                ("Delete", lambda: self.review_selected("delete")),
            )
        ):
            actions.addWidget(self.button(label, callback), index // 5, index % 5)
        layout.addLayout(actions)
        self.browser_search.returnPressed.connect(self.refresh_browser)
        self.browser_filter.currentIndexChanged.connect(self.refresh_browser)

    def refresh_browser(self) -> None:
        if self.repo:
            state, search, offset = (
                self.browser_filter.currentData(),
                self.browser_search.text(),
                self._browser_offset,
            )
            repo = self.repo
            self.run_aux(
                lambda: repo.list_samples(state=state, search=search, limit=200, offset=offset),
                self._browser_ready,
            )

    def browser_page(self, direction: int) -> None:
        self._browser_offset = max(0, self._browser_offset + direction * 200)
        self.refresh_browser()

    def _browser_ready(self, records: list[dict[str, Any]]) -> None:
        self._browser_rows = records
        self.browser_table.setRowCount(len(records))
        for row, sample in enumerate(records):
            values = [
                Path(sample["path"]).name if sample["path"] else sample["id"][:10],
                sample["training_text"],
                f"{sample['duration'] or 0:.2f}",
                sample["language"],
                sample["category"],
                sample["state"],
                sample["wer"],
                sample["cer"],
                sample["speaker_similarity"],
            ]
            for column, value in enumerate(values):
                self.browser_table.setItem(
                    row, column, QTableWidgetItem("—" if value is None else str(value))
                )
        self.browser_page_label.setText(f"Page {self._browser_offset // 200 + 1} • {len(records)} rows")

    def _browser_selected(self) -> None:
        row = self.browser_table.currentRow()
        if not 0 <= row < len(self._browser_rows):
            return
        sample = self._browser_rows[row]
        self._selected_sample = sample
        self.browser_transcript.setPlainText(sample["training_text"])
        self.browser_details.setPlainText(
            json.dumps(
                {
                    key: sample[key]
                    for key in (
                        "spoken_text",
                        "original_text",
                        "asr_text",
                        "wer",
                        "cer",
                        "speaker_similarity",
                        "reasons",
                        "parameters",
                        "metrics",
                        "model_id",
                    )
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        self.waveform.clear()
        if sample["path"]:
            sample_id, path = sample["id"], sample["path"]

            def loaded(result):
                if self._selected_sample and self._selected_sample["id"] == sample_id:
                    self.waveform.set_peaks(result["peaks"], result["duration"])

            self.run_aux(lambda: prepare_waveform(path), loaded)

    def play_selected(self) -> None:
        if self._selected_sample and self._selected_sample["path"]:
            self.play(self._selected_sample["path"])

    def review_selected(self, action: str) -> None:
        if not self.repo or self.runner.busy:
            self.statusBar().showMessage("Stop generation before changing dataset records")
            return
        rows = sorted({index.row() for index in self.browser_table.selectionModel().selectedRows()})
        ids = [self._browser_rows[row]["id"] for row in rows if row < len(self._browser_rows)]
        if not ids:
            return
        reason, ok = QInputDialog.getText(
            self, f"{action.title()} {len(ids)} recording(s)", "Review reason (required):"
        )
        if not ok or not reason.strip():
            return
        if (
            action == "delete"
            and QMessageBox.question(
                self,
                "Delete dataset records",
                "Soft-delete selected records? Audio and source references will remain on disk.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        repo = self.repo
        self.run_task(
            lambda: [repo.review(identifier, action, reason) for identifier in ids],
            lambda _: self.refresh_browser(),
        )

    def edit_selected(self) -> None:
        if self.repo and self._selected_sample:
            identifier, text = self._selected_sample["id"], self.browser_transcript.toPlainText()
            repo = self.repo
            self.run_task(lambda: repo.edit_training_text(identifier, text), lambda _: self.refresh_browser())

    def revalidate_selected(self) -> None:
        if self.repo and self._selected_sample:
            from ..quality import QualityController, revalidate_sample

            try:
                config = self.configuration()
            except Exception as exc:
                self._error(str(exc))
                return
            identifier, repo = self._selected_sample["id"], self.repo
            self.run_task(
                lambda: revalidate_sample(repo, identifier, QualityController(config.quality), config.voice),
                lambda _: self.refresh_browser(),
            )

    def random_review(self) -> None:
        if self.repo:
            repo = self.repo

            def task():
                import random

                records = list(repo.iter_samples(state="ACCEPTED"))
                return random.sample(records, min(10, len(records)))

            self.run_aux(task, self._browser_ready)
