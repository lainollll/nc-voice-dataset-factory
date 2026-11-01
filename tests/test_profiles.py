from __future__ import annotations

from dataclasses import replace


def test_saved_voice_profile_restores_its_synthesis_settings(tmp_path, config, qtbot):
    from dataclasses import asdict
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow

    settings = replace(config.synthesis, temperature=0.63, seed=77)
    voice = replace(config.voice, model_version="mock-v1", synthesis_settings=asdict(settings))
    repo = Repository(tmp_path / "project")
    repo.save_voice(voice)
    window = MainWindow()
    qtbot.addWidget(window)
    window.repo = repo
    window.saved_voices.addItem(voice.name, repo.voices()[0])
    window.saved_voices.setCurrentIndex(1)
    assert window.configuration().synthesis.temperature == 0.63
    assert window.configuration().synthesis.seed == 77
