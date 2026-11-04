from __future__ import annotations

from dataclasses import replace
import threading
from unittest.mock import Mock

from PySide6.QtWidgets import QMessageBox
import pytest


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.metrics_timer.stop()
    window.training_controller.root = tmp_path / "training"
    window.show()
    yield window
    if hasattr(window, "training_inspector") and window.training_inspector.busy:
        qtbot.waitUntil(lambda: not window.training_inspector.busy, timeout=5000)


def project_with_accepted(tmp_path, config, count=2):
    from nc_factory.domain import TextItem
    from nc_factory.storage import Repository

    repo = Repository(tmp_path / "project")
    repo.save_config(config)
    job = repo.create_job(config, [TextItem(f"Fixture sentence number {i}.", f"Fixture sentence number {i}.")
                                   for i in range(count)])
    for sample in repo.list_samples(job_id=job):
        repo.update_sample(sample["id"], state="ACCEPTED", duration=60.0)
    return repo


def mark_environment_ready(window):
    from nc_factory.ui.training_controller import training_python

    python = training_python(window.training_controller.root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture only; never executed")
    (window.training_controller.root / "environment_ready.json").write_text("{}")


def wait_for_source(window, qtbot):
    qtbot.waitUntil(lambda: not window.training_source_timer.isActive()
                   and not window.training_inspector.busy, timeout=5000)


def test_training_empty_state_explains_missing_recordings(window):
    mark_environment_ready(window)
    window.open_page("LFM3 Fine-tuning")
    assert not window.training_start.isEnabled()
    assert "project" in window.training_source_info.text().lower()
    assert "Ready" in window.training_environment.text()
    assert "fine-tun" in window.guide.instructions.toPlainText().lower()
    assert not window.training_details.isExpanded()


def test_training_uses_accepted_source_summary_and_preserves_request(window, qtbot, tmp_path, config):
    window.repo = project_with_accepted(tmp_path, config)
    mark_environment_ready(window)
    window.open_page("LFM3 Fine-tuning")
    wait_for_source(window, qtbot)
    assert "2 accepted" in window.training_source_info.text()
    assert "2.0 minutes" in window.training_source_info.text()
    assert config.voice.name in window.training_source_info.text()
    assert window.training_start.isEnabled()
    captured = []
    window.training_controller.start_training = captured.append
    window.training_steps.setValue(345)
    window._start_training()
    assert len(captured) == 1
    assert captured[0].project_root == window.repo.root
    assert captured[0].dataset_root is None and captured[0].max_steps == 345


def test_training_environment_must_be_ready_before_start(window, qtbot, tmp_path, config):
    window.repo = project_with_accepted(tmp_path, config)
    window.open_page("LFM3 Fine-tuning")
    wait_for_source(window, qtbot)
    assert not window.training_start.isEnabled()
    assert "Set up" in window.training_environment.text()
    start = Mock()
    window.training_controller.start_training = start
    window._start_training()
    start.assert_not_called()


def test_training_discard_stale_export_validation(window, qtbot, monkeypatch, tmp_path):
    from nc_factory.ui import training_page
    from nc_factory.ui.training_data import TrainingSourceSummary

    release = threading.Event()
    entered = threading.Event()
    owner_thread = threading.get_ident()
    worker_threads = []

    def inspect(repo, export):
        worker_threads.append(threading.get_ident())
        if export.name == "first":
            entered.set()
            release.wait(3)
        return TrainingSourceSummary(True, export.name, 3, 180, "Fixture voice", "en")

    monkeypatch.setattr(training_page, "inspect_training_source", inspect)
    mark_environment_ready(window)
    window.training_source.setCurrentIndex(1)
    window.training_dataset.setText(str(tmp_path / "first"))
    window.open_page("LFM3 Fine-tuning")
    qtbot.waitUntil(entered.is_set)
    window.training_dataset.setText(str(tmp_path / "second"))
    assert not window.training_start.isEnabled()
    release.set()
    wait_for_source(window, qtbot)
    assert "second" in window.training_source_info.text()
    assert "first" not in window.training_source_info.text()
    assert window.training_start.isEnabled()
    assert all(thread != owner_thread for thread in worker_threads)


def test_training_force_stop_requires_explicit_confirmation(window, monkeypatch):
    force = Mock()
    window.training_controller.force_stop = force
    window.training_controller._stage = "training"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.No)
    window._confirm_force_stop()
    force.assert_not_called()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    window._confirm_force_stop()
    force.assert_called_once()
    window.training_controller._stage = ""


def test_training_progress_uses_real_stage_and_steps(window):
    window._training_progress({"stage": "encoding", "message": "Encoding recordings"})
    assert "Encoding" in window.training_stage.text()
    assert window.training_progress.maximum() == 0
    window._training_progress({"stage": "training", "step": 12, "total": 200})
    assert window.training_progress.maximum() == 200
    assert window.training_progress.value() == 12
    assert "12" in window.training_stage.text()
    window._training_finished(1)
    assert window.training_details.isExpanded()


def test_training_source_inspector_accepts_only_current_validated_samples(tmp_path, config):
    from nc_factory.ui.training_data import inspect_training_source

    repo = project_with_accepted(tmp_path, config, 3)
    records = repo.list_samples()
    repo.update_sample(records[0]["id"], integrity_hold=1)
    repo.update_sample(records[1]["id"], training_revision=1)
    summary = inspect_training_source(repo, None)
    assert summary.ready and summary.accepted == 1 and summary.seconds == 60
    assert summary.voice == config.voice.name and summary.language == "en"


def test_training_source_inspector_rejects_missing_export_without_creating_it(tmp_path):
    from nc_factory.ui.training_data import inspect_training_source

    missing = tmp_path / "does not exist"
    summary = inspect_training_source(None, missing)
    assert not summary.ready and not missing.exists()
    assert summary.message


def test_preview_custom_text_and_settings_are_snapshotted_on_worker(window, qtbot, reference, monkeypatch, tmp_path):
    from nc_factory.tts import MockTTS
    from nc_factory.ui import voice_page

    monkeypatch.setattr(voice_page, "APPLICATION_ROOT", tmp_path / "standalone")
    window.reference_path.setText(str(reference))
    window.permission.setChecked(True)
    window.engine.setCurrentIndex(window.engine.findData("mock"))
    window.preview_text.setPlainText("A custom preview to compare these settings.")
    window.synthesis_spins["temperature"].setValue(0.63)
    owner = threading.get_ident()
    observed = []
    release = threading.Event()

    class Backend(MockTTS):
        def synthesize(self, text, voice, settings, seed):
            release.wait(3)
            observed.append((text, voice, settings, threading.get_ident()))
            return super().synthesize(text, voice, settings, seed)

    monkeypatch.setattr(voice_page, "create_tts", lambda settings: Backend())
    played = []
    window.play = played.append
    window.test_voice()
    window.synthesis_spins["temperature"].setValue(1.4)
    window.preview_text.setPlainText("Edited while the old preview was running.")
    release.set()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    assert observed[0][0] == "A custom preview to compare these settings."
    assert observed[0][2].temperature == 0.63 and observed[0][3] != owner
    assert observed[0][1].reference_sha256
    assert window._reference_digest == ""  # Previewing does not silently save the voice.
    assert len(played) == 1 and "standalone" in played[0]
    assert window.preview_replay.isEnabled()
    assert window.repo is None
    window._replay_voice_preview()
    assert len(played) == 2 and played[0] == played[1]


def test_preview_refuses_changed_saved_reference_without_rehashing(window, qtbot, reference, monkeypatch, tmp_path):
    from nc_factory.tts import MockTTS
    from nc_factory.ui import voice_page

    monkeypatch.setattr(voice_page, "APPLICATION_ROOT", tmp_path)
    window.reference_path.setText(str(reference))
    window.permission.setChecked(True)
    window._reference_digest = "wrong saved hash"
    backend = MockTTS()
    backend.synthesize = Mock()
    monkeypatch.setattr(voice_page, "create_tts", lambda settings: backend)
    window.test_voice()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    backend.synthesize.assert_not_called()
    assert "hash" in window.preview_status.text().lower()
    assert window._reference_digest == "wrong saved hash"


def test_preview_language_updates_default_but_keeps_custom_text(window):
    window.language.setCurrentIndex(window.language.findData("sv"))
    assert "Hej" in window.preview_text.toPlainText()
    window.preview_text.setPlainText("Keep these exact words.")
    window.language.setCurrentIndex(window.language.findData("en"))
    assert window.preview_text.toPlainText() == "Keep these exact words."


def test_preview_settings_keep_existing_configuration_bindings(window, tmp_path, config):
    from nc_factory.storage import Repository

    window.repo = Repository(tmp_path / "project")
    selected = replace(config, synthesis=replace(config.synthesis, temperature=0.61, exaggeration=0.72,
                                                cfg_weight=0.45, top_p=0.92, seed=99))
    window.apply_config(selected)
    voice, settings = window.voice_configuration()
    assert settings == window.configuration().synthesis == selected.synthesis
    assert voice.reference_sha256 == config.voice.reference_sha256
    assert window.synthesis_spins["top_p"].value() == 0.92


def test_preview_does_not_start_during_training(window, monkeypatch):
    from nc_factory.ui import voice_page

    backend = Mock()
    monkeypatch.setattr(voice_page, "create_tts", backend)
    window.training_controller._stage = "training"
    window.test_voice()
    backend.assert_not_called()
    assert not window.runner.busy
    assert "training" in window.preview_status.text().lower()
    window.training_controller._stage = ""


def test_speech_sliders_and_saved_spin_values_stay_in_sync(window, tmp_path, config):
    from nc_factory.storage import Repository

    window.synthesis_sliders["temperature"].setValue(730)
    assert window.synthesis_spins["temperature"].value() == 0.73
    window.repo = Repository(tmp_path / "project")
    window.apply_config(replace(config, synthesis=replace(config.synthesis, cfg_weight=0.63)))
    assert window.synthesis_sliders["cfg_weight"].value() == 630
    assert window.configuration().synthesis.cfg_weight == 0.63


def test_explicit_reference_import_clears_hash_even_when_path_is_unchanged(window, reference, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    window.reference_path.setText(str(reference))
    window._reference_digest = "old saved digest"
    window.permission.setChecked(True)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (str(reference), ""))
    window.choose_reference()
    assert window._reference_digest == ""
    assert not window.permission.isChecked()


def test_quick_preview_carries_wav_and_permission_without_creating_project(window, reference, monkeypatch):
    window.quick_start.reference.setText(str(reference))
    window.quick_start.language.setCurrentIndex(window.quick_start.language.findData("sv"))
    window.quick_start.permission.setChecked(True)
    window.synthesis_spins["temperature"].setValue(0.67)
    monkeypatch.setattr(window, "find_preview_models", lambda: None)
    window.quick_start.preview_requested.emit()
    assert window.navigation.currentItem().text() == "Voice Manager"
    assert window.reference_path.text() == str(reference)
    assert window.permission.isChecked() and window.language.currentData() == "sv"
    assert window.synthesis_spins["temperature"].value() == 0.67
    assert window.repo is None and window.pipeline is None and not window.runner.busy


def test_training_setup_finish_rechecks_pending_source_without_claiming_training_completed(
    window, qtbot, tmp_path, config,
):
    window.repo = project_with_accepted(tmp_path, config)
    window.training_controller._stage = "setup"
    window.training_controller.busy_changed.emit(True)
    window.open_page("LFM3 Fine-tuning")
    assert not window.training_start.isEnabled()
    mark_environment_ready(window)
    window.training_controller._stage = ""
    window.training_controller.busy_changed.emit(False)
    window.training_controller.finished.emit(0)
    wait_for_source(window, qtbot)
    assert window.training_start.isEnabled()
    assert "tools ready" in window.training_stage.text().lower()
    assert not window.training_adapter.isEnabled()


@pytest.mark.parametrize("page", ["LFM3 Fine-tuning", "Voice Manager"])
def test_guided_pages_fit_narrow_window_without_horizontal_scrolling(window, qtbot, page):
    window.resize(1000, 780)
    window.open_page(page)
    qtbot.wait(20)
    assert window.pages.currentWidget().horizontalScrollBar().maximum() == 0


def test_preview_primary_action_is_visible_without_scrolling_at_normal_size(window, qtbot):
    from PySide6.QtCore import QPoint

    window.resize(1250, 850)
    window.open_page("Voice Manager")
    qtbot.wait(20)
    assert window.preview_button.mapTo(window, QPoint(0, window.preview_button.height())).y() < window.height() - 30
    assert "preview" in window.guide.step_title.text().lower()


def test_closing_waits_for_owned_source_inspector(window, qtbot, tmp_path, config, monkeypatch):
    from nc_factory.ui import training_page
    from nc_factory.ui.training_data import TrainingSourceSummary

    entered, release = threading.Event(), threading.Event()

    def inspect(*args):
        entered.set()
        release.wait(3)
        return TrainingSourceSummary(False, "No training will start")

    monkeypatch.setattr(training_page, "inspect_training_source", inspect)
    window.repo = project_with_accepted(tmp_path, config)
    window.open_page("LFM3 Fine-tuning")
    qtbot.waitUntil(entered.is_set)
    window.close()
    assert window._closing and window.isVisible()
    release.set()
    qtbot.waitUntil(lambda: not window.training_inspector.busy and not window.isVisible(), timeout=5000)


def test_preview_failure_closes_backend_and_restores_controls(window, qtbot, reference, monkeypatch, tmp_path):
    from nc_factory.tts import MockTTS
    from nc_factory.ui import voice_page

    backend = MockTTS()
    backend.synthesize = Mock(side_effect=RuntimeError("fixture preview failure"))
    backend.close = Mock()
    monkeypatch.setattr(voice_page, "create_tts", lambda settings: backend)
    monkeypatch.setattr(voice_page, "APPLICATION_ROOT", tmp_path)
    window.reference_path.setText(str(reference))
    window.permission.setChecked(True)
    window.engine.setCurrentIndex(window.engine.findData("mock"))
    window.test_voice()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    backend.close.assert_called_once()
    assert "fixture preview failure" in window.preview_status.text()
    assert window.preview_button.isEnabled()


def test_source_summary_rejects_mixed_voice_jobs(tmp_path, config):
    from nc_factory.domain import TextItem
    from nc_factory.ui.training_data import inspect_training_source

    repo = project_with_accepted(tmp_path, config)
    other = replace(config, voice=replace(config.voice, id="another-voice", name="Another voice"))
    job = repo.create_job(other, [TextItem("A separate voice sample.", "A separate voice sample.")])
    repo.update_sample(repo.list_samples(job_id=job)[0]["id"], state="ACCEPTED", duration=2)
    summary = inspect_training_source(repo, None)
    assert not summary.ready and "one voice" in summary.message


def test_editing_reference_requires_new_permission_before_preview(window, reference, monkeypatch):
    from nc_factory.ui import voice_page

    window.reference_path.setText(str(reference))
    window.permission.setChecked(True)
    other = reference.with_name("another-reference.wav")
    other.write_bytes(reference.read_bytes())
    window.reference_path.setText(str(other))
    assert not window.permission.isChecked()
    backend = Mock()
    monkeypatch.setattr(voice_page, "create_tts", backend)
    window.test_voice()
    backend.assert_not_called()
    assert not window.runner.busy


def test_sliders_remain_draggable_in_narrow_window(window, qtbot):
    window.resize(1000, 780)
    window.open_page("Voice Manager")
    qtbot.wait(20)
    assert all(slider.width() >= 80 for slider in window.synthesis_sliders.values())


def test_quick_preview_discovers_multilingual_model_when_leaving_english(
    window, qtbot, reference, tmp_path, monkeypatch,
):
    from nc_factory.quick_setup import LocalModels
    from nc_factory.ui import voice_page

    window.engine.setCurrentIndex(window.engine.findData("english"))
    window.model_path.setText(str(tmp_path / "english-weights"))
    multilingual = tmp_path / "multilingual-weights"
    discover = Mock(return_value=LocalModels("multilingual-v3", multilingual, None))
    monkeypatch.setattr(voice_page, "discover_models", discover)
    window.quick_start.reference.setText(str(reference))
    window.quick_start.language.setCurrentIndex(window.quick_start.language.findData("sv"))
    window.quick_start.permission.setChecked(True)
    window.quick_start.preview_requested.emit()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    discover.assert_called_once()
    assert window.engine.currentData() == "multilingual-v3"
    assert window.model_path.text() == str(multilingual)


def test_auxiliary_diagnostics_do_not_repeat_completed_dataset_inspection(window, qtbot, tmp_path, config):
    window.repo = project_with_accepted(tmp_path, config)
    window.open_page("LFM3 Fine-tuning")
    wait_for_source(window, qtbot)
    start = Mock(wraps=window.training_inspector.start)
    window.training_inspector.start = start
    window.aux_runner.start(lambda: None)
    qtbot.waitUntil(lambda: not window.aux_runner.busy, timeout=5000)
    assert not window.training_source_timer.isActive()
    start.assert_not_called()
