from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


def test_training_setup_keeps_arrow_compatible_with_numpy_one(tmp_path):
    from packaging.requirements import Requirement
    from nc_factory.ui.training_controller import environment_commands

    # A missing Arrow constraint previously let pip choose 26 with NumPy 1.26,
    # passing pip check but failing the native Arrow import at setup verification.
    requirements = {}
    for _, arguments in environment_commands(tmp_path):
        if arguments[:3] != ["-m", "pip", "install"] or "--index-url" in arguments:
            continue
        for value in arguments[3:]:
            requirement = Requirement(value)
            requirements[requirement.name] = requirement.specifier
    assert "1.26.4" in requirements["numpy"]
    assert "pyarrow" in requirements, "Training must constrain Arrow's NumPy ABI compatibility"
    assert "20.0.0" in requirements["pyarrow"]
    assert "26.0.0" not in requirements["pyarrow"]


def notebook_fixture() -> dict:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["Notebook fixture"]} for _ in range(30)]
    for index, source in {
        4: "import torch\nimport transformers\nimport peft\n",
        6: 'model_id = "LiquidAI/LFM3-Audio-2B"\nmodel = object()\nprocessor = object()\n',
        16: "def create_collate_fn(prompt):\n    return lambda examples: examples\ncollate_fn = create_collate_fn(SYSTEM_PROMPT)\nbatch = collate_fn([train_dataset[0], train_dataset[1]])\n",
        18: "from peft import LoraConfig, get_peft_model\npeft_config = LoraConfig(r=16)\nmodel = get_peft_model(model, peft_config)\n",
    }.items():
        cells[index] = {
            "cell_type": "code",
            "metadata": {},
            "source": source.splitlines(True),
            "execution_count": None,
            "outputs": [],
        }
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_notebook_adapter_uses_own_data_and_executable_cells():
    from nc_factory.training import adapt_notebook

    notebook = adapt_notebook(notebook_fixture())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for code in codes:
        ast.parse(code)
    assert notebook["metadata"]["nc_factory"]["source_revision"]


def test_download_rejects_notebook_with_unexpected_bytes(monkeypatch):
    from nc_factory import training

    class Response:
        content = json.dumps(notebook_fixture()).encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(training.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="checksum"):
        training.fetch_notebook()
    monkeypatch.setattr(training, "NOTEBOOK_SHA256", hashlib.sha256(Response.content).hexdigest())
    assert training.fetch_notebook()["nbformat"] == 4


def test_training_preparation_snapshots_current_project_and_never_changes_runtime(
    tmp_path, config, monkeypatch
):
    from dataclasses import replace
    from nc_factory import training
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=2),
        [
            TextItem(
                "Please bring a book from the library today.", "Please bring a book from the library today."
            ),
            TextItem(
                "The weather across the ocean changes every season.",
                "The weather across the ocean changes every season.",
            ),
        ],
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    monkeypatch.setattr(training, "fetch_notebook", notebook_fixture)
    request = training.TrainingRequest(project_root=repo.root, max_steps=5)
    prepared = training.prepare_training(request, tmp_path / "training")
    assert prepared.run_root.is_relative_to(tmp_path / "training" / "runs")
    saved = json.loads((prepared.run_root / "training_request.json").read_text(encoding="utf-8"))
    assert saved["max_steps"] == 5 and saved["dataset_root"] == "dataset"
    assert (prepared.run_root / "dataset" / "manifest.jsonl").is_file()
    assert (prepared.run_root / "load_lfm3_dataset.py").is_file()
    assert prepared.notebook.is_file() and prepared.runner.is_file()
    assert repo.get_job(job)["status"] == "COMPLETED"
    assert not (tmp_path / "training" / "env").exists()
    second = training.prepare_training(request, tmp_path / "training")
    assert second.run_root != prepared.run_root
    # Exercise the adapted load/encode cells, preserving the export's split IDs.
    import sys
    import numpy as np
    from nc_factory import lfm3
    from nc_factory.training_notebook import CONFIG_CELL, DATA_CELL, ENCODE_CELL

    monkeypatch.chdir(prepared.run_root)
    monkeypatch.setitem(sys.modules, "load_lfm3_dataset", lfm3)
    with pytest.raises(RuntimeError, match="separate"):
        exec(CONFIG_CELL, {})
    saved["generation_prefix"] = str(tmp_path / "generation-env")
    (prepared.run_root / "training_request.json").write_text(json.dumps(saved), encoding="utf-8")

    class Processor:
        def encode_audio(self, audio):
            return np.zeros((2, 12), dtype=np.int64)

    namespace = {"processor": Processor()}
    exec(CONFIG_CELL, namespace)
    exec(DATA_CELL, namespace)
    train_ids = namespace["raw_ds"]["train"]["id"]
    validation_ids = namespace["raw_ds"]["validation"]["id"]
    exec(ENCODE_CELL, namespace)
    assert namespace["train_dataset"]["id"] == train_ids
    assert namespace["eval_dataset"]["id"] == validation_ids
    assert namespace["SYSTEM_PROMPT"] == "Perform TTS in English using Test voice's voice."
    assert not set(train_ids) & set(validation_ids)


def test_training_preparation_rejects_invalid_requests_and_empty_project(tmp_path):
    from nc_factory.training import TrainingRequest, prepare_training

    with pytest.raises(ValueError, match="project|export"):
        prepare_training(TrainingRequest(), tmp_path / "training")
    with pytest.raises(ValueError, match="steps"):
        prepare_training(TrainingRequest(dataset_root=tmp_path, max_steps=0), tmp_path / "training")
    assert not (tmp_path / "training" / "runs").exists()


def test_training_controller_setup_is_async_and_failure_never_marks_ready(qtbot, tmp_path, monkeypatch):
    import sys
    from nc_factory.ui import training_controller

    monkeypatch.setattr(
        training_controller,
        "environment_commands",
        lambda root: [(sys.executable, ["-u", "-c", "print('fixture setup failure'); raise SystemExit(3)"])],
    )
    controller = training_controller.TrainingController(tmp_path, lambda: True)
    messages = []
    controller.log.connect(messages.append)
    with qtbot.waitSignal(controller.finished, timeout=5000) as done:
        controller.setup()
        assert controller.busy
    assert done.args[0] == 3 and not controller.ready
    assert any("fixture setup failure" in message for message in messages)


def test_runner_executes_fixture_cells_in_order_and_honors_stop(tmp_path, monkeypatch):
    from nc_factory.training_runner import execute_notebook

    monkeypatch.chdir(tmp_path)
    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["value = 40\n"]},
            {
                "cell_type": "code",
                "source": ["from pathlib import Path\nPath('answer.txt').write_text(str(value + 2))\n"],
            },
        ]
    }
    path = tmp_path / "fixture.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    execute_notebook(path)
    assert (tmp_path / "answer.txt").read_text() == "42"
    (tmp_path / "STOP").touch()
    with pytest.raises(InterruptedError):
        execute_notebook(path)


def test_training_blocks_generation_and_close_stops_owned_process(qtbot, tmp_path, monkeypatch):
    import sys
    from nc_factory.ui import training_controller
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(
        training_controller,
        "environment_commands",
        lambda root: [(sys.executable, ["-u", "-c", "import time; print('ready'); time.sleep(30)"])],
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.training_controller.root = tmp_path
    window.training_controller.setup()
    qtbot.waitUntil(lambda: window.training_controller.process.processId() > 0)
    assert window.training_controller.busy
    window.start_job()
    assert window.pipeline is None and not window.runner.busy
    window.close()
    qtbot.waitUntil(lambda: not window.training_controller.busy, timeout=10000)
    qtbot.waitUntil(lambda: not window.isVisible())


@pytest.mark.parametrize("stop", [False, True])
def test_training_process_reports_progress_and_saves_or_stops(qtbot, tmp_path, monkeypatch, stop):
    import sys
    from nc_factory.training import PreparedTraining, TrainingRequest
    from nc_factory.ui import training_controller

    root = tmp_path / "training"
    run = root / "runs" / "fixture"
    run.mkdir(parents=True)
    (root / "environment_ready.json").write_text("{}", encoding="utf-8")
    script = run / "fixture.py"
    script.write_text(
        "import json,time\nfrom pathlib import Path\n"
        "print('NC_TRAINING_EVENT '+json.dumps(dict(stage='training',step=1,total=2)),flush=True)\n"
        + ("while not Path('STOP').exists(): time.sleep(0.02)\n" if stop else "")
        + "p=Path('artifacts/adapter'); p.mkdir(parents=True)\n"
        "(p/'adapter_config.json').write_text('{}')\n"
        "(p/'adapter_model.safetensors').write_bytes(b'fixture')\n"
        + ("raise SystemExit(130)\n" if stop else ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_controller, "training_python", lambda root: Path(sys.executable))
    monkeypatch.setattr(
        training_controller, "prepare_training", lambda *args: PreparedTraining(run, script, script)
    )
    controller = training_controller.TrainingController(root, lambda: True)
    events, statuses = [], []
    controller.event.connect(events.append)
    controller.status.connect(statuses.append)
    with qtbot.waitSignal(controller.finished, timeout=5000) as done:
        controller.start_training(TrainingRequest())
        qtbot.waitUntil(lambda: bool(events))
        if stop:
            controller.stop()
    assert done.args[0] == (130 if stop else 0)
    assert events[0]["step"] == 1
    assert not controller.busy
    assert (run / "artifacts" / "adapter" / "adapter_model.safetensors").is_file()
    assert ("Stopped" if stop else "completed") in statuses[-1]
