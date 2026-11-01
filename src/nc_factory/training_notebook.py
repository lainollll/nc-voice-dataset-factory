"""Adapt only the pinned cookbook's data/configuration cells for an NC run."""

from __future__ import annotations

import ast
from typing import Any

COOKBOOK_REVISION = "3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232"
NOTEBOOK_NAME = "lfm3_audio_tts_sft_with_transformers.ipynb"
NOTEBOOK_URL = (
    f"https://github.com/Liquid4All/cookbook/blob/{COOKBOOK_REVISION}/finetuning/notebooks/{NOTEBOOK_NAME}"
)

CONFIG_CELL = """from pathlib import Path
import json
import sys
import os
RUN_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "training_request.json").is_file())
config = json.loads((RUN_ROOT / "training_request.json").read_text(encoding="utf-8"))
if Path(sys.prefix).resolve() == Path(config["generation_prefix"]).resolve():
    raise RuntimeError("Select the separate LFM3 training environment, not the generation environment.")
os.environ["WANDB_DISABLED"] = "true"
OUTPUT_DIR = RUN_ROOT / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
def check_stop():
    if (RUN_ROOT / "STOP").exists():
        raise InterruptedError("Training stopped by user")
def event(stage, **fields):
    print("NC_TRAINING_EVENT " + json.dumps(dict(stage=stage, **fields)), flush=True)
check_stop()
"""

DATA_CELL = """import io
import numpy as np
import soundfile as sf
from load_lfm3_dataset import load_lfm3_dataset, encode_for_lfm3
check_stop()
event("dataset", message="Validating the exported train/validation split")
raw_ds = load_lfm3_dataset(RUN_ROOT / config["dataset_root"], cache_dir=RUN_ROOT / "cache")
if not len(raw_ds["train"]):
    raise ValueError("The export has no training recordings")
language_name = {"en": "English", "sv": "Swedish"}[config["language"]]
SYSTEM_PROMPT = f"Perform TTS in {language_name} using {config['voice_name']}'s voice."
print(f"Training voice: {config['voice_name']} ({language_name})", flush=True)
"""

ENCODE_CELL = """check_stop()
event("encoding", message="Encoding your recordings with the LFM3 codec")
class StoppableEncoder:
    def encode_audio(self, audio):
        check_stop()
        return processor.encode_audio(audio)
encoded_ds = encode_for_lfm3(raw_ds, StoppableEncoder())
train_dataset, eval_dataset = encoded_ds["train"], encoded_ds["validation"]
check_stop()
print(f"Train: {len(train_dataset)}; validation: {len(eval_dataset)}", flush=True)
"""

TRAIN_CELL = """from transformers import Trainer, TrainingArguments, TrainerCallback
check_stop()
class NCProgress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        event("training", step=state.global_step, total=state.max_steps)
        if (RUN_ROOT / "STOP").exists():
            control.should_save = True
            control.should_training_stop = True
        return control
    def on_log(self, args, state, control, logs=None, **kwargs):
        event("training", step=state.global_step, total=state.max_steps, metrics=logs or {})
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    max_steps=config["max_steps"],
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=min(10, config["max_steps"]),
    logging_steps=1,
    eval_strategy="steps" if len(eval_dataset) else "no", eval_steps=50,
    save_strategy="steps", save_steps=min(50, config["max_steps"]), save_total_limit=2,
    bf16=True, remove_unused_columns=False, report_to="none", dataloader_num_workers=0,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=eval_dataset if len(eval_dataset) else None,
    data_collator=collate_fn, callbacks=[NCProgress()])
event("training", step=0, total=config["max_steps"])
trainer.train()
event("saving", message="Saving the trained LoRA adapter and processor")
trainer.save_model(str(OUTPUT_DIR / "adapter"))
processor.save_pretrained(str(OUTPUT_DIR / "adapter"))
if (RUN_ROOT / "STOP").exists():
    raise InterruptedError("Stopped safely; adapter and available checkpoints saved")
event("completed", message=str(OUTPUT_DIR / "adapter"))
"""


def adapt_notebook(source: dict[str, Any]) -> dict[str, Any]:
    def upstream(index: int, marker: str) -> str:
        try:
            cell = source["cells"][index]
            code = "".join(cell["source"])
            if cell["cell_type"] != "code" or marker not in code:
                raise ValueError("Unexpected upstream notebook layout")
            return code
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("Unexpected upstream notebook layout") from exc

    collator = upstream(16, "def create_collate_fn").partition("\nbatch =")[0]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LFM3 fine-tuning on your NC dataset\n",
                f"Adapted from the [Liquid cookbook notebook]({NOTEBOOK_URL}).\n",
                "The app manages a separate environment. Your saved train/validation splits are preserved.\n",
                "Start from the LFM3 Fine-tuning card. Stop requests save at a training-step boundary.\n",
                "Outputs are LoRA adapters and checkpoints; no speaker-similarity result is claimed.\n",
            ],
        }
    ]
    for code in (
        CONFIG_CELL,
        upstream(4, "import torch"),
        "if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():\n"
        '    raise RuntimeError("This notebook requires a CUDA GPU with bfloat16 support")\n'
        'check_stop()\nevent("model", message="Loading the LFM3 model and codec")\n',
        upstream(6, "LiquidAI/LFM3-Audio-2B"),
        DATA_CELL,
        ENCODE_CELL,
        collator,
        upstream(18, "LoraConfig"),
        TRAIN_CELL,
    ):
        ast.parse(code)
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": code.splitlines(True),
                "execution_count": None,
                "outputs": [],
            }
        )
    return {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "LFM3 training", "name": "python3", "language": "python"},
            "language_info": {"name": "python"},
            "nc_factory": {"source_revision": COOKBOOK_REVISION, "source_url": NOTEBOOK_URL},
        },
    }
