def test_large_generated_corpus_has_bounded_editable_preview(qtbot):
    from nc_factory.domain import TextItem
    from nc_factory.text import diversity_statistics
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    items = [
        TextItem(f"This is a generated example for topic {i}.", f"This is a generated example for topic {i}.")
        for i in range(1001)
    ]
    window._preview_ready((items, diversity_statistics(items), []))
    assert window.text_preview.document().blockCount() == 500
    assert len(window._preview_items) == 1001
    assert "first 500" in window.diversity_label.text()
