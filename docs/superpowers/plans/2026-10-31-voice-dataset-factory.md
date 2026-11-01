# NC Voice Dataset Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Implement and test all five approved phases of the standalone voice dataset application.

**Architecture:** A typed Python backend owns project storage, jobs, audio, QC and export. PySide6 submits blocking operations to workers. DeepSeek supplies scoped implementation proposals and independent reviews; the lead inspects, integrates and executes tests.

**Tech Stack:** Python 3.11, PySide6, SQLite, NumPy/scipy/soundfile/librosa, Chatterbox, faster-whisper, optional datasets/speaker embeddings, pytest.

**Spec:** `docs/superpowers/specs/2026-10-31-voice-dataset-factory-design.md`

## Global Constraints

- Standalone app in `H:\TTS training factory`; do not edit NeuralCompanion or global dependencies.
- No commits, pushes, remotes, automatic model downloads or overwriting accepted files.
- One GPU synthesis worker, CPU/int8 ASR default, 24 kHz mono PCM16 output.
- Exact synthesis text is immutable; edited training text requires revalidation.
- Every enabled QC gate must pass; mocks cannot establish real voice acceptance.
- Persist immutable job configuration and every attempt; recover files and DB together.
- Approval covers all phases and execution with DeepSeek; no additional planning approval.

## Review Focus

- Missing or changed reference material must block generation (phase 1 test).
- Process failure between file publication and DB update must not overwrite/reaccept files (phase 2 test).
- Transcript edit after acceptance must invalidate export eligibility without changing audio (phase 3 test).
- Near-duplicate transcripts must remain in one split (phase 4 test).
- Closing during inference must preserve workers until cooperative completion (phase 5 Qt test).

## Task 1 — functional voice-to-dataset pipeline

Files: `pyproject.toml`, `requirements*.txt`, `scripts/setup.ps1`, `src/nc_factory/{domain,storage,audio,tts,pipeline,export,cli}.py`, `src/nc_factory/ui/{window,workers,theme}.py`, `tests/test_phase1.py`.

Interfaces: `ProjectConfig`, `VoiceProfile`, `SynthesisSettings`, `QualitySettings`, `SampleState`; `Repository(root: Path)` owns SQLite; `TTSBackend.synthesize(text, voice, settings, seed) -> AudioResult`; `Pipeline.run(job_id)`; `export_dataset(repo, destination, options) -> ExportReport`.

- [x] Write tests for reference hash changes, atomic no-clobber WAVs, mock/native adapter parameter compatibility, accepted CSV mapping and 10-clip basic pipeline.
- [x] Run tests and confirm failure before application modules exist.
- [x] Implement domain/config validation, storage, WAV I/O, native and explicit mock backends, sequential batch, CSV export and functional Qt shell with worker bridge.
- [x] Run phase 1 tests and offscreen Qt startup; record real backend availability separately.
- [x] DeepSeek task review and resolve actionable findings.

## Task 2 — durable automation and diverse text

Files: `src/nc_factory/{text,pipeline,storage}.py`, `src/nc_factory/ui/window.py`, `tests/test_phase2.py`.

Interfaces: `generate_texts(count, language, weights, seed, min_words, max_words) -> list[TextItem]`; import/local-LLM text sources use same records; pipeline control uses `threading.Event`; job DB owns status/accepted counts and attempt budget.

- [x] Write and run failing tests for weighted bilingual generation/dedup, import sanitization, retry replenishment, pause/stop/resume and interrupted publication recovery.
- [x] Implement indexed diversity/duplicates, local LLM/imports, immutable job snapshots, project writer lock, recovery/retry/budgets and UI controls.
- [x] Run all phase 1/2 tests and inspect persistence after restart.
- [x] DeepSeek task review and resolve findings.

## Task 3 — quality assurance and dataset review

Files: `src/nc_factory/{quality,audio,storage,pipeline}.py`, `src/nc_factory/ui/{window,waveform}.py`, `tests/test_phase3.py`.

Interfaces: `QualityController.check(path, reference_text, language, voice) -> QualityResult`; `ASRBackend.transcribe(path, language)`; speaker calibration stores model/reference/calibration hashes; `Repository.review/edit/requeue` audit actions.

- [x] Write/run failing tests for WER/CER normalization, silence/clipping/nonfinite/bad WAV, inaccurate ASR/retries, calibration, edits and manual review.
- [x] Implement audio QC, faster-whisper lazy local loader, independent speaker calibration, acceptance gates and searchable review/playback/waveform.
- [x] Run complete suite and mocked 10-clip verification pipeline.
- [x] DeepSeek task review and resolve findings.

## Task 4 — complete training exports

Files: `src/nc_factory/export.py`, `examples/load_lfm3_dataset.py`, `tests/test_phase4.py`, export UI.

Interfaces: `ExportOptions(formats, validation_fraction, seed, include_rejected)`; manifest validation and grouped splits; HF `DatasetDict` with embedded WAV bytes, `normalized_text`, original fields and train/validation subsets.

- [x] Write/run failing tests for deterministic grouped splitting, CSV/LJSpeech exact mapping, rejected exclusion, missing/hash-changed WAVs and failure atomicity.
- [x] Implement all formats, provenance, snapshot integrity, optional HF save/load and official notebook-compatible loader/converter.
- [x] Run complete suite, export/reload smoke, and optional HF tests with installed export dependencies.
- [x] DeepSeek task review and resolve findings.

## Task 5 — production operation, GUI and delivery

Files: `src/nc_factory/{diagnostics,cli}.py`, all UI pages, `README.md`, `examples/project.json`, `scripts/{setup,start}.ps1`, runtime locks/compatibility notes, `tests/test_phase5.py`.

- [x] Write/run failing tests for Qt worker close/cancel, pagination, configuration round trip, disk estimation and model-unavailable errors.
- [x] Complete all seven pages with real service bindings; bounded work, GPU memory/status, throughput/ETA, storage warnings, logging and startup diagnostics.
- [x] Run all tests, compile/import checks, offscreen Qt workflow and CLI 10-recording smoke; inspect actual exported artifacts.
- [x] DeepSeek whole-application review and fix correctness issues with regression tests.
- [x] Record dependency lock/test evidence and every unverified hardware/model requirement; deliver setup/start commands and real acceptance instructions.

## Rulings

- This is an empty non-Git workspace: work directly in the user-selected project folder. Git worktree/commit scripts do not apply; no Git repository will be created just to satisfy workflow machinery.
- DeepSeek connector agents return proposals, not workspace edits. Lead applies reviewed proposals using file patches and runs tests locally.
- New dedicated environment creation via the project setup script is authorized by implementation approval; never modify an existing unrelated environment.
- Model downloads and real generation remain explicit user actions until a permitted voice is selected; do not search unrelated folders for private voices.

## Final real-output gate

- [x] User waived the exact 1,000-file benchmark and requested stopping. Verified 817 accepted English recordings (82.572 minutes), published export, 735/82 grouped split and complete embedded WAV/transcript pairing through both saved HF data and the standalone LFM3 loader. Original job status remains STOPPED; no claim of reaching 1,000.
