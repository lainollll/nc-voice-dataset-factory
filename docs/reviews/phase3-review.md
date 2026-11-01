# Independent Review: Phase 3 Quality/ASR/Speaker/Transcript Source

**Method:** Static reading of the four supplied modules (`quality.py`, `speaker.py`, `transcript.py`, plus the `revalidate_sample` in the same file as `quality.py`). I did not run tests; statements about runtime behavior are derived from the code. Cross-checked against the reported 24-test results.

**Not re-reporting as bugs** (working-as-intended per the passing-test description): mock-ASR label `MOCK-ASR-NOT-RECOGNITION`, empty `errors=0` no-download guarantee for `FasterWhisperASR`, calibration class-overlap rejection, reference-hash binding, threshold-tamper detection, `asr_error`/`speaker_error` → `needs_review`.

---

## 1. Spec compliance

### Offline model guarantees
- **`FasterWhisperASR.__init__`**: enforces `asr_model_path` is a dir, presence of `model.bin`, `config.json`, `tokenizer.json`; message explicitly says tokenizer fallback downloads are disabled. ✔
- **`FasterWhisperASR.transcribe`**: lazy `WhisperModel(..., local_files_only=True, ...)`. ✔ **But see Bug Q1** — dependency pinning is not enforced.
- **`SpeakerEmbedding.__init__`**: requires `hyperparams.yaml`; hashes `.ckpt`/`.yaml` into `model_id`. ✔
- **`SpeakerEmbedding.embed`**: requires `fetch_config`/`local_strategy` kwargs on `pretrained_from_hparams`, uses `LocalStrategy.COPY` + `FetchConfig(allow_network=False)`. ✔ (Defensive; the API reliance is real — see Bug S1.)
- **Calibration load** binds to `model_id` + `reference_sha256` and self-recomputes; a tampered `threshold` is detected. ✔
- **`calibrate_reference`**: rejects non-distinct recordings vs. reference by sha. ✔ Defense in depth.

### Scoring validity
- **WER/CER**: DP `edit_distance` with O(min(n,m)) memory, ratio semantics (`0/0→0`, `denom=0,hyp≠0→1.0`). Standard. ✔ Reasonable.
- **`normalize_for_comparison`**: NFC → casefold → domain `text.normalize_text` (falls back on `ValueError` only), word extraction `[^\W_]+`. The fallback path is only triggered on `ValueError`; the docstring says "when it is not usable (e.g. for ASR instruction markers)" — only `ValueError` from `normalize_text` (see Bug T1). ✔ functionally; **docstring slightly overstates**.
- **Calibration separation**: `lower_positive > upper_negative` required; midpoint threshold saved with `method` disclaimer "held-out verification recommended". Honest. ✔
- **`SpeakerVerifier.compare`**: cosine of L2-normalized embeddings; threshold from validated calibration. ✔

### Enabled gates
- `verify_asr` false → `metrics["asr_check"]="disabled"`, ASR block skipped. ✔
- `speaker_enabled` false → `metrics["speaker_check"]="disabled"`, speaker block skipped. ✔
- `check_repetition` false → `repeated_audio_correlation` and `repeated_audio` reason both skipped. ✔
- `check_abrupt_edges` gate honored. ✔
- `settings.validate()` earlier in `prepare` and implicit re-validation in `voice.validate()` inside both verifiers. ✔

### Exact text immutability
- `revalidate_sample` re-reads `sample["training_text"]` and `sample["language"]` from the repo under `writer_lock()`; audio path re-hashes via `sha256_file` vs. stored `sha256`; `validated_revision = training_revision` on pass else `-1`. ✔
- The controller does not mutate `reference_text`; `normalize_for_comparison` operates on copies. ✔
- **Caveat (Q3):** `revalidate_sample` writes `reasons` verbatim even if pre-existing reasons from the previous validation mode differ; but that's a semantic note, not an immutability violation.

### Loaded API signatures vs. source
- `ASRBackend.transcribe(path: Path, language: str)` — `MockASR` and `FasterWhisperASR` both match. ✔
- `QualityController(settings, asr=None, speaker=None)` and `check(path, reference_text, language, voice)` used consistently in `revalidate_sample`. ✔
- `SpeakerVerifier.compare(path) -> tuple[float, float]` matches call site `similarity, threshold = self.speaker.compare(...)`. ✔
- `asdict(asr)` / `asdict(s)` require dataclasses → `ASRResult` and `QualitySettings` are `@dataclass`. q.py: not visible; the asdict call assumes it — the source shows `ASRResult` is a dataclass; `QualitySettings` is imported from `.domain` and assumed dataclass. Not verifiable from excerpt but load-bearing (Q4).
- `pretrained_from_hparams(..., cls=, source=, savedir=, overrides=, run_opts=, local_strategy=, fetch_config=)` — only `fetch_config` is signature-checked; the rest are trusted by name (Bug S1).

---

## 2. Quality verdict: **Ship with two low-risk fixes; one medium-risk fix recommended before Phase 4**

Correctness of the core scoring pipeline (audio gate → optional ASR → optional speaker → immutable revalidation) is sound. The offline guarantee is enforced by explicit pre-checks plus runtime kwargs. Two defects (Q1, S1) are **quality/reproducibility** issues rather than correctness regressions today; one (S2) is a latent correctness issue that will manifest once a real model is wired up.

---

## 3. Concrete, source-supported findings

### Q1 (Medium): `FasterWhisperASR` accepts an unpinned `faster-whisper` version at runtime
- **Trigger:** any environment with a different `faster-whisper` installed than the one tested. The identity `f"faster-whisper/{version}/..."` is *reported* but never *enforced*, and the `WhisperModel` call still succeeds with incompatible versions (or silently changed defaults such as `vad_filter`).
- **Evidence:** `quality.py::FasterWhisperASR.transcribe` — `identity = f"faster-whisper/{importlib.metadata.version('faster-whisper')}/{...}"`; no comparison against an expected version, and `prepare()` never pins it.
- **Fix (small):** add `asr_library_version: str` to `QualitySettings` and raise in `FasterWhisperASR.__init__` when `importlib.metadata.version("faster-whisper") != settings.asr_library_version`. Alternatively compare at transcribe-time against a module-level `EXPECTED_FASTER_WHISPER` and raise `RuntimeError` on mismatch.

### Q2 (Low): `no_speech_probability` is reported as `max` over segments but `avg_logprob` as `mean`
- **Trigger:** ASR results with a mix of speech/silence segments. The pair `(avg_logprob, no_speech_probability)` is not from the same segment, so the `-1.2` / `0.6` review gate is checking marginally inconsistent evidence.
- **Evidence:** `float(np.mean([s.avg_logprob for s in segments]))` vs. `max((s.no_speech_prob for s in segments), default=1.0)`.
- **Fix (small):** use `max(s.no_speech_prob for s in segments)` for the gate as intended, but also record `avg_no_speech_probability` (mean) in `metrics` via `asdict(asr)` — i.e., mirror the aggregation. Or, cleaner, gate on `max` semantics and document that.

### Q3 (Low, semantic): `revalidate_sample` writes rejection reasons but does not clear stale `wer`/`cer`/`speaker_similarity` on transitions
- **Trigger:** sample previously passed (`wer` set), then revalidated and rejected because e.g. `reference_text` was edited to mismatch. `update_sample(..., wer=result.wer, ...)` writes the new `None`, which is correct — but if the caller passes `None` through `update_sample` as "leave unchanged" (not visible), stale values survive. **Only if** `update_sample` treats `None` as no-op.
- **Evidence:** `update_sample(sample_id, ..., wer=result.wer, cer=result.cer, ...)` — the `None` handling in `storage.update_sample` is not in the excerpt; flagged for review, not asserted.
- **Fix (small):** read `storage.update_sample`; if `None` means "keep", add a sentinel or explicit `wer=result.wer if result.wer is not None else 0.0`.

### T1 (Low): `normalize_for_comparison` fallback is only `ValueError`, but docstring claims broader coverage
- **Trigger:** `text.normalize_text` raising anything other than `ValueError` (e.g., `RuntimeError` from a missing resource) escapes the guard and propagates as `asr_error_review_required` masking a normalization bug.
- **Evidence:** `except ValueError: normalized = unicodedata.normalize("NFC", text).casefold()` — docstring: "falls back to plain punctuation cleanup without raising when it is not usable".
- **Fix (small):** either narrow the docstring to `ValueError` (preferred) or broaden to `except Exception` with logging, but broadening invites silent data loss. **Recommend narrowing the docstring.**

### S1 (Medium): `SpeakerEmbedding.embed` trusts all kwargs of `pretrained_from_hparams` except `fetch_config`
- **Trigger:** SpeechBrain version where `local_strategy`s values, or the `overrides`/`run_opts` shapes changed. Only `fetch_config` is checked via `inspect.signature`, so a version bump that renames `savedir` or changes `overrides` semantics will fail later with a confusing `TypeError` inside library code, not our guarded error message.
- **Evidence:** `if "fetch_config" not in inspect.signature(pretrained_from_hparams).parameters: raise ...` — one keyword checked; `cls=source=savedir=overrides=run_opts=local_strategy=` all assumed.
- **Fix (small):** build a required-param set, e.g. `req
