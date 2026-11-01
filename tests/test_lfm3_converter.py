from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def test_converter_uses_official_codes_tolist_contract():
    from datasets import Audio, Dataset, DatasetDict, Features, Value
    from nc_factory.lfm3 import encode_for_lfm3

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(24000, dtype="float32"), 24000, format="WAV")
    features = Features({"audio": Audio(decode=False), "normalized_text": Value("string")})
    data = DatasetDict(
        train=Dataset.from_dict(
            {
                "audio": [{"bytes": buffer.getvalue(), "path": "x.wav"}],
                "normalized_text": ["Read this sentence."],
            },
            features=features,
        )
    )

    class Codes:
        def __array__(self, *args):
            raise TypeError("Simulated CUDA tensor cannot convert through numpy")

        def tolist(self):
            return [[1] * 12, [2] * 12]

    class Processor:
        def encode_audio(self, waveform):
            assert waveform.ndim == 1 and len(waveform) == 24000
            return Codes()

    result = encode_for_lfm3(data, Processor())
    assert result["train"][0]["codes"] == [[1] * 12, [2] * 12]
    assert result["train"][0]["normalized_text"] == "Read this sentence."
