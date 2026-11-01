from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QDoubleSpinBox,
    QHeaderView,
    QGroupBox,
    QVBoxLayout,
    QSizePolicy,
)


class TextPageMixin:
    def _text_page(self) -> None:
        from ..corpus import CATEGORIES

        layout = self.page(
            "Text Generator",
            "Prepare sentences in your selected voice's language, then review them on Batch Generator before creating audio.",
        )
        source_form = QFormLayout()
        source_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.text_count = QSpinBox()
        self.text_count.setRange(1, 100000)
        self.text_count.setValue(1000)
        self.text_method = QComboBox()
        for label, value in (
            ("Built-in corpus — offline templates", "builtin"),
            ("OpenAI-compatible LLM", "local-llm"),
            ("Import a text file", "imported"),
        ):
            self.text_method.addItem(label, value)
        source_form.addRow("Sentence source", self.text_method)
        source_form.addRow("Preview sentence count", self.text_count)
        self.text_count.setToolTip(
            "How many sentences to prepare now. Set the accepted-audio target on Batch Generator."
        )
        layout.addLayout(source_form)
        self.text_source_help = QLabel()
        self.text_source_help.setWordWrap(True)
        layout.addWidget(self.text_source_help)
        self.text_language_note = QLabel()
        self.text_language_note.setWordWrap(True)
        self.text_language_note.setObjectName("subtitle")
        layout.addWidget(self.text_language_note)
        limits = QFormLayout()
        limits.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.min_words, self.max_words = QSpinBox(), QSpinBox()
        for spin, value in ((self.min_words, 6), (self.max_words, 36)):
            spin.setRange(1, 200)
            spin.setValue(value)
        self.topics = QLineEdit()
        self.topics.setPlaceholderText("Optional: everyday life, travel, science")
        self.llm_endpoint = QLineEdit("http://127.0.0.1:1234/v1")
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        self.llm_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.llm_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.llm_model.setMinimumContentsLength(12)
        self.llm_model.lineEdit().setPlaceholderText("Select a discovered model, or type a model ID")
        self.allow_remote = QCheckBox("Allow a remote server")
        self.allow_remote.setToolTip("Enable only to contact a server outside this computer, including a LAN server.")
        limits.addRow("Minimum words", self.min_words)
        limits.addRow("Maximum words", self.max_words)
        limits.addRow("Custom topics", self.topics)
        layout.addLayout(limits)
        self.llm_options = QGroupBox("OpenAI-compatible text model")
        llm_layout = QVBoxLayout(self.llm_options)
        server_help = QLabel(
            "Start your OpenAI-compatible server and enter its base URL including /v1. "
            "The model list fills automatically; choose a chat/instruct model for writing sentences. "
            "Voice Manager controls the speech model."
        )
        server_help.setWordWrap(True)
        llm_layout.addWidget(server_help)
        server_form = QFormLayout()
        server_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        server_form.addRow("Server base URL", self.llm_endpoint)
        server_form.addRow("Text model", self.llm_model)
        llm_layout.addLayout(server_form)
        self.refresh_models_button = self.button("Refresh models", self._refresh_llm_models)
        llm_layout.addWidget(self.refresh_models_button)
        llm_layout.addWidget(self.allow_remote)
        self.llm_models_status = QLabel("Models will load when you choose this source.")
        self.llm_models_status.setWordWrap(True)
        self.llm_models_status.setTextFormat(Qt.TextFormat.PlainText)
        llm_layout.addWidget(self.llm_models_status)
        layout.addWidget(self.llm_options)
        self.builtin_options = QGroupBox("Built-in sentence mix")
        builtin_layout = QVBoxLayout(self.builtin_options)
        category_help = QLabel(
            "Relative weights control the mix of template categories. Weight 20 requests twice as "
            "many sentences as weight 10; zero disables a category. The weights need not total 100. "
            "These controls apply to the built-in corpus."
        )
        category_help.setWordWrap(True)
        builtin_layout.addWidget(category_help)
        self.category_table = QTableWidget(len(CATEGORIES), 2)
        self.category_table.setHorizontalHeaderLabels(["Category", "Relative weight"])
        self.category_table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.category_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.category_weights = {}
        for index, category in enumerate(CATEGORIES):
            self.category_table.setItem(index, 0, QTableWidgetItem(category.replace("_", " ").title()))
            weight = QDoubleSpinBox()
            weight.setRange(0, 100)
            weight.setValue(100 / len(CATEGORIES))
            self.category_weights[category] = weight
            self.category_table.setCellWidget(index, 1, weight)
        builtin_layout.addWidget(self.category_table)
        self.generate_preview_button = self.button("Generate preview", self.generate_preview, True)
        layout.addWidget(self.generate_preview_button)
        layout.addWidget(self.button("Import TXT / CSV / JSONL / JSON…", self.import_preview))
        layout.addWidget(self.button("Review on Batch Generator", lambda: self.open_page("Batch Generator")))
        self.diversity_label = QLabel("No text generated yet; preview appears on Batch Generator.")
        self.diversity_label.setWordWrap(True)
        layout.addWidget(self.diversity_label)
        layout.addWidget(self.builtin_options)
        self._llm_models_key: tuple[str, bool] | None = None
        self._llm_models_pending = False
        self._llm_models_timer = QTimer(self)
        self._llm_models_timer.setSingleShot(True)
        self._llm_models_timer.setInterval(500)
        self._llm_models_timer.timeout.connect(self._run_pending_model_discovery)
        self.text_method.currentIndexChanged.connect(self._text_source_changed)
        self.language.currentIndexChanged.connect(self._text_source_changed)
        self.llm_endpoint.textChanged.connect(self._llm_settings_changed)
        self.allow_remote.toggled.connect(self._llm_settings_changed)
        self.navigation.currentRowChanged.connect(self._text_page_selected)
        self.aux_runner.finished.connect(self._run_pending_model_discovery)
        self._text_source_changed()

    @Slot()
    def _text_source_changed(self) -> None:
        method = self.text_method.currentData()
        self.builtin_options.setVisible(method == "builtin")
        self.llm_options.setVisible(method == "local-llm")
        for control in (
            self.text_count,
            self.min_words,
            self.max_words,
            self.topics,
            self.generate_preview_button,
        ):
            control.setEnabled(method != "imported")
        self.text_source_help.setText(
            {
                "builtin": "Built-in corpus combines bundled English/Swedish sentence templates with names, places, "
                "objects and other word choices. It works offline without an LLM. Duplicates are filtered. "
                "Its variety is finite, so large requests or narrow word/category settings can exhaust it. "
                "Custom topics replace topic placeholders where a template uses one.",
                "local-llm": "Your selected text model writes new sentences about your custom topics. "
                "Generated sentences are normalized, deduplicated and filtered by the word limits. "
                "Use this source for more varied text. The server must be running.",
                "imported": "Import TXT (one sentence per line), CSV with a text column, or JSON/JSONL "
                "strings or objects with a text field. "
                "Sentences are normalized and duplicate/invalid rows are skipped with a report. "
                "Preview count, word limits, topics and category weights do not filter imported text.",
            }.get(method, "Choose a sentence source to prepare your text.")
        )
        self.text_language_note.setText(
            f"Voice language: {self.language.currentText()}. Preview count prepares text now; "
            "Batch Generator sets how many accepted recordings or minutes to create. "
            "Review/edit the prepared sentences there."
        )
        if method == "local-llm":
            self._schedule_llm_models()
        else:
            self._llm_models_pending = False
            self._llm_models_timer.stop()

    @Slot(int)
    def _text_page_selected(self, _row: int) -> None:
        if self.navigation.currentItem().text() == "Text Generator":
            self._schedule_llm_models()
        else:
            self._llm_models_timer.stop()
            self._llm_models_pending = False

    @Slot()
    def _llm_settings_changed(self) -> None:
        self._llm_models_key = None
        current = self.llm_model.currentText()
        self.llm_model.clear()
        self.llm_model.setEditText(current)
        self._schedule_llm_models()

    def _schedule_llm_models(self) -> None:
        key = (self.llm_endpoint.text().strip(), self.allow_remote.isChecked())
        if (
            self.text_method.currentData() == "local-llm"
            and self.navigation.currentItem().text() == "Text Generator"
            and key != self._llm_models_key
        ):
            self._llm_models_pending = True
            self._llm_models_timer.start()

    @Slot()
    def _refresh_llm_models(self) -> None:
        self._llm_models_timer.stop()
        self._llm_models_pending = True
        self._run_pending_model_discovery()

    @Slot()
    def _run_pending_model_discovery(self) -> None:
        if not self._llm_models_pending or self._llm_models_timer.isActive():
            return
        if (
            self._closing
            or not self.isVisible()
            or self.text_method.currentData() != "local-llm"
            or self.navigation.currentItem().text() != "Text Generator"
        ):
            self._llm_models_pending = False
            return
        if self.aux_runner.busy or self._aux_pending is not None:
            self.llm_models_status.setText("Waiting for the current background read before fetching models…")
            return
        from ..text import list_llm_models

        key = (self.llm_endpoint.text().strip(), self.allow_remote.isChecked())
        self._llm_models_pending = False
        self.llm_models_status.setText("Fetching available models from the server…")

        def task():
            try:
                return key, list_llm_models(*key), ""
            except Exception as exc:
                return key, [], str(exc)

        self.run_aux(task, self._llm_models_ready)

    def _llm_models_ready(self, result: tuple[tuple[str, bool], list[str], str]) -> None:
        key, models, error = result
        if (
            self._closing
            or self.text_method.currentData() != "local-llm"
            or key != (self.llm_endpoint.text().strip(), self.allow_remote.isChecked())
        ):
            return
        if error:
            self.llm_models_status.setText(
                f"Unable to list models: {error} You can still type a model ID, then retry Refresh models."
            )
            return
        current = self.llm_model.currentText()
        self.llm_model.clear()
        self.llm_model.addItems(models)
        if current:
            self.llm_model.setCurrentText(current)
        self._llm_models_key = key
        message = f"Found {len(models)} models. Choose a chat/instruct model for sentence generation."
        if current and current not in models:
            message += " Your saved/custom ID is preserved; choose a listed ID to change it."
        self.llm_models_status.setText(message)

    def generate_preview(self) -> None:
        from ..text import diversity_statistics, generate_texts, generate_llm_corpus

        count, language = self.text_count.value(), self.language.currentData()
        weights = {key: spin.value() for key, spin in self.category_weights.items()}
        minimum, maximum = self.min_words.value(), self.max_words.value()
        topics = [t.strip() for t in self.topics.text().split(",") if t.strip()]
        method, endpoint, model, remote = (
            self.text_method.currentData(),
            self.llm_endpoint.text(),
            self.llm_model.currentText(),
            self.allow_remote.isChecked(),
        )

        def task():
            if method == "local-llm":
                items = generate_llm_corpus(
                    endpoint,
                    model,
                    count,
                    language,
                    minimum,
                    maximum,
                    ", ".join(topics) or "everyday conversation",
                    remote,
                    self._task_cancel,
                )
            elif method == "imported":
                raise ValueError("Use Import to select your source file")
            else:
                items = generate_texts(count, language, weights, 0, minimum, maximum, topics)
            return items, diversity_statistics(items), []

        self.run_task(task, self._preview_ready)

    def import_preview(self) -> None:
        from ..text import diversity_statistics, import_texts

        path, _ = QFileDialog.getOpenFileName(
            self, "Import source text", filter="Text datasets (*.txt *.csv *.jsonl *.json)"
        )
        language = self.language.currentData()
        if path:
            self.text_method.setCurrentIndex(self.text_method.findData("imported"))

            def task():
                items, warnings = import_texts(Path(path), language)
                return items, diversity_statistics(items), warnings

            self.run_task(task, self._preview_ready)

    def _preview_ready(self, result: tuple) -> None:
        items, stats, warnings = result
        self._preview_items = items
        self.text_preview.setPlainText("\n".join(t.spoken_text for t in items[:500]))
        self.diversity_label.setText(
            f"{len(items)} unique texts • {stats['unique_words']} distinct words • {len(warnings)} skipped rows. {stats['phonetic_coverage']}"
            + (
                " Preview shows the first 500; edits affect shown rows and the remaining texts are retained."
                if len(items) > 500
                else ""
            )
        )
        for warning in warnings[:50]:
            self.batch_log.appendPlainText(warning)
