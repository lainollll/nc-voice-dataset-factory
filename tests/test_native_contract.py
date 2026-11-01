from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.parametrize("engine", ["english", "multilingual-v2", "multilingual-v3"])
def test_native_uses_offline_local_interface_true_sample_rate_and_single_load(
    tmp_path, config, monkeypatch, engine
):
    from nc_factory.tts import NativeChatterbox

    calls = []

    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 22050), dtype="float32")

    class Model:
        sr = 22050

        @classmethod
        def from_local(cls, ckpt_dir, device, t3_model=None):
            calls.append(("load", device, t3_model))
            return cls()

        def generate(
            self,
            text,
            audio_prompt_path=None,
            language_id=None,
            temperature=0.8,
            exaggeration=0.5,
            cfg_weight=0.5,
            repetition_penalty=1.2,
            top_p=1.0,
            min_p=0.05,
        ):
            calls.append(("generate", text, audio_prompt_path, language_id, temperature))
            return Tensor()

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.manual_seed = lambda _: None
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = types.ModuleType("chatterbox.tts" if engine == "english" else "chatterbox.mtl_tts")
    setattr(module, "ChatterboxTTS" if engine == "english" else "ChatterboxMultilingualTTS", Model)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tokenizer = types.ModuleType("chatterbox.models.tokenizers.tokenizer")
    tokenizer.hf_hub_download = lambda **kwargs: pytest.fail("An implicit network fetch was attempted")
    monkeypatch.setitem(sys.modules, tokenizer.__name__, tokenizer)
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "test.safetensors").write_bytes(b"fake local weights for contract test")
    (directory / "Cangjie5_TC.json").write_text("[]", encoding="utf-8")
    settings = replace(config.synthesis, engine=engine, device="cpu", model_dir=str(directory))
    backend = NativeChatterbox(settings)
    first = backend.synthesize("Please bring the red book.", config.voice, settings, 12)
    backend.synthesize("Here is a different sentence.", config.voice, settings, 13)
    assert first.sample_rate == 22050
    assert first.parameters["seed"] == 12
    assert first.model_id.startswith("chatterbox/") and "sha256:" in first.model_id
    assert len([c for c in calls if c[0] == "load"]) == 1
    assert calls[0][2] == (None if engine == "english" else engine.rsplit("-", 1)[-1])
    assert calls[1][3] == (None if engine == "english" else "en")
