from __future__ import annotations

from dataclasses import replace

import pytest


def test_quality_controls_preserve_margin_and_optional_checks(tmp_path, config, qtbot):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow

    cfg = replace(
        config,
        quality=replace(config.quality, trim_margin_ms=200, check_abrupt_edges=False, check_repetition=False),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.repo = Repository(tmp_path / "project")
    window.apply_config(cfg)
    saved = window.configuration()
    assert saved.quality.trim_margin_ms == 200
    assert not saved.quality.check_abrupt_edges
    assert not saved.quality.check_repetition


def test_llm_corpus_duplicate_responses_have_bounded_attempts(monkeypatch):
    from nc_factory.domain import TextItem
    from nc_factory import text

    attempts = []

    def repetitive(*args, **kwargs):
        attempts.append(1)
        return [
            TextItem("The sun is shining over the old harbour.", "The sun is shining over the old harbour.")
        ]

    monkeypatch.setattr(text, "local_llm_texts", repetitive)
    with pytest.raises(ValueError, match="unique|enough"):
        text.generate_llm_corpus("http://127.0.0.1:1234/v1", "model", 10, "en")
    assert len(attempts) == 4
