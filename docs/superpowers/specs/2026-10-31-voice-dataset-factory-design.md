# NC Voice Dataset Factory — standalone application design

Status: proposed for user review. No application implementation has started.

## Intent and scope

Build a reusable Windows 11 desktop application that converts a permitted reference
voice and diverse English or Swedish text into verified, individually paired speech
recordings for TTS training. Target Python 3.11, PySide6, native Chatterbox and an
NVIDIA RTX 4090. Implement all five phases in the supplied request sequentially;
the first working pipeline is a milestone, not the final deliverable.

The full supplied request remains the acceptance specification. This document
records architecture, operational behavior, validation boundaries, and proposed
implementation decisions. It does not reduce the feature scope.

## Approaches and selected direction

1. Standalone PySide6 application in `H:\TTS training factory` with independent
   projects, jobs, and adapter interfaces. Protects NeuralCompanion and permits
   testing without loading the live companion. Future integration uses explicit
   adapters. **Selected by the user on 2026-10-31.**
2. Implement inside NeuralCompanion. Could reuse its voice-library UI and runtime,
   but its repository is absent here and training jobs would share its runtime.
   Not selected.

Use the standalone direction without editing existing NeuralCompanion files,
external runtimes, global packages, existing virtual environments, caches, secrets,
or remotes. No commits or pushes. A setup script will create a dedicated environment
only when invoked; dependencies must not replace global NeuralCompanion packages.

## Inspection findings and verified upstream contracts

The workspace was empty when inspected. Python 3.11, PySide6 6.11.2, and NumPy
2.4.6 are visible to the 3.11 interpreter, but PyTorch, Chatterbox, faster-whisper,
soundfile, librosa, scipy, pytest, and datasets are not. FFmpeg is available.
The RTX 4090 reports 24,564 MiB total memory and about 8,122 MiB free at inspection;
free memory is transient and must be measured again at job startup.

Sources inspected on 2026-10-31:

- [Chatterbox repository](https://github.com/resemble-ai/chatterbox)
- [English adapter source](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts.py)
- [Multilingual adapter source](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/mtl_tts.py)
- [Chatterbox dependency metadata](https://github.com/resemble-ai/chatterbox/blob/master/pyproject.toml)
- [Official Liquid training notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_audio_tts_sft_with_transformers.ipynb)

Verified current source supports `ChatterboxTTS.from_pretrained(device=...)` and
`ChatterboxMultilingualTTS.from_pretrained(device=..., t3_model="v3")`; multilingual
generation takes `text`, `language_id`, and `audio_prompt_path`. Both generation
methods expose temperature, exaggeration, CFG weight, repetition penalty, min-p,
and top-p. English and Swedish appear in the multilingual language table. Both
return watermarked audio tensors and expose the output sample rate as `sr`.

The current upstream package metadata identifies version 0.1.7 and pins PyTorch
and torchaudio 2.6.0 for Python 3.11. Current metadata and installed NumPy differ;
do not combine them in the global environment. An immutable upstream revision and
a resolved, tested application dependency lock must be recorded during setup.
Do not label proposed dependency versions as tested before validation.

The Liquid notebook reads `audio["bytes"]` through soundfile and uses
`normalized_text` as the authoritative text. It resamples audio to 24 kHz and calls
`processor.encode_audio()` before constructing training conversations. Exported
WAVs are training inputs, not precomputed codec tokens. Training is outside this
application; a loader/converter replaces the notebook's dataset-loading section
and preserves its codec preprocessing and collator. Keep Chatterbox and notebook
training dependencies separate because their Transformers versions differ.

## Architecture and ownership

Use typed Python with inward-facing protocols; backend modules never import Qt.

| Module | Responsibility |
| --- | --- |
| `domain` | Validated project, voice, utterance, job, attempt, QC, and export models |
| `storage` | SQLite migrations, repositories, project locks, atomic audio and config writes |
| `text` | English/Swedish normalization, corpus/templates, imports, local LLM, duplicate index |
| `tts` | Common adapter contract, native English/multilingual, documented-service adapter, mocks |
| `audio` | Decode, resample, trim, WAV encoding, signal metrics, fingerprints |
| `quality` | faster-whisper, WER/CER, optional independent speaker verification |
| `jobs` | Persistent orchestration, retry policy, accepted targets, recovery, progress events |
| `export` | CSV/LJSpeech/JSONL, HF, Liquid loader, deterministic splits, integrity validation |
| `ui` | Seven pages, worker bridge, table model, waveform, playback, editable configuration |
| `app` | Composition, logging, CLI smoke/diagnostics commands, application lifecycle |

TTS adapters return audio, true sample rate, effective parameters, model identity,
and timing. ASR returns transcript, confidence indicators, language, model identity,
and timing. Optional adapters are imported lazily and report actionable dependency
errors. Mock TTS/ASR run the same pipeline and are visibly labeled test backends;
synthetic tones must never be presented as verified cloned speech.

## Project, voice, and persistence model

Each project has a versioned JSON configuration and SQLite database, with isolated
audio, original-audio, rejected-attempt, reference, export, and log directories.
Use WAL, foreign keys, short transactions, and one connection per worker/thread.
Use a project-level exclusive writer lock with stale-owner recovery so two app
instances cannot generate or edit the same project concurrently.

Persist projects/config revisions, voice profiles, utterances, jobs, attempts,
quality results, review actions, and export snapshots. Record timestamps, seeds,
dependency versions, exact backend/model revision, reference hash, effective
synthesis settings, original and spoken text, category, language, and provenance.
An immutable job snapshot prevents UI configuration changes from changing a running
job. Utterance IDs and attempt IDs are stable; each attempt has its own filename.

Voice Manager imports a reference WAV and can save multiple named local profiles.
Store voice name, language, source, reference location/hash, model identity, settings,
and explicit permission confirmation. Copy imported references into the project
only as an explicit import operation; never delete the user's source. Recheck the
hash before generation and refuse missing/changed references until resolved.

Support external saved profiles when a connected installation exposes a verified
interface. The official native library does not expose a general saved-voice
registry or HTTP service: do not invent either. A service connection requires its
documented API/OpenAPI contract and explicit configuration; show unavailable
capabilities and a reason rather than offering a nonfunctional connection.
Define a voice-provider protocol for future NeuralCompanion integration.

## Text production and diversity

Provide authored sentence banks and grammatical templates in both languages for
all requested categories. Deterministic seeded generation combines compatible
phrases and varies topics, names, places, numbers, and sentence structures. Weights
are validated and normalized; report achieved category distribution. Built-in
content is finite: surface exhaustion honestly and offer imports or local LLM
generation rather than padding with counters or endlessly recycling a sentence.

Imported UTF-8 TXT/CSV/JSONL maps fields explicitly and reports rejected rows.
OpenAI-compatible local LLM requests use a configured model and loopback endpoint,
bounded request size, timeout, cancellation, and strict sentence-list parsing.
Remote endpoints require an explicit user configuration. Strip/reject formatting,
instructions, role markers, and malformed outputs before any synthesis.

Keep original text separate from the exact normalized text sent to the adapter.
Normalization is versioned and language-aware, with conservative abbreviation and
number expansion and configurable length limits. Default estimated durations are
3–15 seconds; actual duration is validated after synthesis. Ambiguous dates,
addresses, and number forms stay visible for preview/editing rather than being
silently guessed.

Use canonical hashes for exact duplicates and an indexed token/character-shingle
similarity search for near duplicates. Track lexical coverage, length distribution,
category balance, and phonetic coverage using a documented phoneme provider when
available. Without one, label character/phonetic proxies explicitly. Do not claim
a template corpus guarantees naturalness or complete phonetic coverage.

## Generation, state, and recovery

One synthesis worker owns one model instance. GUI operations submit bounded work;
model loads, inference, imports, waveform decoding, ASR, exports, and large queries
run outside the Qt main thread. Use queued signals, immutable progress snapshots,
and cooperative cancellation; never terminate a QThread during inference.

Sample states follow `PENDING → GENERATING → GENERATED → VALIDATING → ACCEPTED`,
with `REJECTED` for failed quality checks and `FAILED` for operational errors.
Every transition is transactional and creates an attempt/event record. Rejected
or failed samples can be explicitly retried with a new attempt and seed; preserve
prior results and rejected audio for review. Never overwrite accepted audio.

Pause finishes the current sample to a durable boundary and stops dequeuing work.
Resume uses the same job snapshot. Stop cooperatively finishes the current stage,
stores durable state, and releases the worker; restart/resume is explicit. Persist
pause/stop intent. On restart, reconcile interrupted rows, published files, and
partial writes under the exclusive project lock: validate existing generated audio
before reuse; regenerate incomplete attempts with new attempt paths. Never reset
accepted rows during recovery.

Use temporary files on the destination volume, flush/fsync, close, publish
atomically, then persist hashes and the durable transition. File/database updates
cannot share one transaction: recovery must reconcile both crash directions.
Temporary/orphan files are quarantined/reported, not silently adopted or deleted.

Targets count accepted recordings or accepted duration. Generate replacement text
as needed after rejection, with per-sample retry limits, a bounded total-attempt
budget, and a visible exhausted/needs-attention result. Exhaustion is not success.
ETA uses measured accepted throughput and rejection rate. Persist enough timing
to provide meaningful restart progress.

## GPU and quality policy

Probe CUDA and memory on a worker. Reuse a single model, inference mode, and one
generation at a time. Default faster-whisper to CPU/int8 to avoid VRAM contention;
GPU verification is opt-in and serial. Report GPU identity, utilization when
available, free/allocated/reserved memory, load failures, and actual timings.
On CUDA OOM, release temporary tensors, clear unused cache where appropriate,
record failure, and use a bounded retry/pause policy. Do not terminate other GPU
processes, silently switch models, or claim additional memory was reclaimed.

Audio checks cover decoding, emptiness, finite samples, true sample rate, duration,
clipping, low level, silence ratio, edge energy/cutoff risk, repeated segments, and
exact/near-duplicate audio. Use proper resampling, mono PCM16 output at 24 kHz by
default, and optional conservative trimming with margins. Retain original audio
when requested. Do not perform aggressive loudness changes or watermark removal.
Cutoff and repetition heuristics expose reasons and measurements for review.

ASR compares against the spoken source; it never replaces it. Versioned comparison
normalization handles Unicode, punctuation/case, numbers, and language conventions.
Store ASR text, WER/CER, recognition confidence, repeated/missing-text indicators,
and thresholds. Start with configurable WER 0.10, and expose stricter presets.
Unreliable recognition goes to review with reasons. Every enabled quality gate
must pass before automatic acceptance; disabled gates are recorded as not run.

Independent speaker verification is optional and CPU-capable. Select a supported
pretrained embedding model after inspecting its current API. Require a calibration
set and store its provenance/score distribution/threshold; no invented universal
similarity cutoff. An enabled but unconfigured or failing verifier blocks automatic
acceptance and explains why. The reference is not a sufficient negative calibration
set by itself.

## Desktop interaction

Use a consistent dark theme, readable typography, sidebar navigation, clear status
and error messages, and persistent project selection. The pages are Dashboard,
Voice Manager, Text Generator, Batch Generator, Quality Control, Dataset Browser,
and Export. Each page operates on the same repositories and services.

Dashboard reports selected voice, accepted/rejected counts, accepted duration,
job state, GPU metrics, disk estimate, throughput, and ETA. Voice Manager provides
reference import, permission confirmation, saved profiles, model/language/settings,
reference playback, and worker-based Test Voice.

Text Generator includes count, duration/length bounds, category weights, imports,
local model configuration, preview/editing, and diversity statistics. Batch page
provides accepted count/duration targets, retries, disk warnings, current sentence,
audio, progress, speed, errors, and start/pause/resume/stop/retry actions.

Quality Control edits thresholds, ASR and speaker calibration settings, and links
to flagged examples. Dataset Browser uses a paginated/searchable/filterable model,
playback and a decimated waveform, source/ASR text, duration, language, category,
WER/CER/similarity, reasons, effective settings, and single/batch review actions.
Manual approval is recorded as an override with actor/time/reason. Transcript edits
invalidate old transcript-verification results and require revalidation; retain
the original synthesis text in provenance. Delete requires explicit user action
and cannot affect imported source references. Regeneration creates a new attempt.

Offer random accepted-sample listening review before export and record review
status. Export page shows snapshot selection, format, destination, split seed,
validation issues, progress, and final counts/duration. Closing with work active
requests cooperative stop and waits asynchronously without losing worker objects.

## Exports and training integration

Export an immutable accepted-only snapshot by default; rejected data is included
only through an explicit option and kept separate from training splits. Never
write into an existing export silently. Stage output, validate it, then publish
the completed directory without touching the working dataset.

Produce the requested folder layout: `audio/`, `metadata.csv`,
`metadata_ljspeech.csv`, `train.jsonl`, `validation.jsonl`, `manifest.jsonl`,
`dataset_info.json`, `project_config.json`, `rejected/`, and `logs/`.
CSV contains filename, text, normalized_text, language, split. LJSpeech contains
file ID, original text, normalized text and rejects/escapes unsupported delimiters
according to the format rather than corrupting rows.

Deterministic splitting groups exact and near-duplicate transcripts before
assigning groups, preventing related transcripts from crossing splits. Stable
IDs, seed, and split algorithm version are stored. Validate final transcript/audio
mapping, decodeability, sample rate/channels/subtype, hashes, duplicates, missing
files, split disjointness, record counts, and accepted duration. Fail export on
integrity errors instead of producing a success report.

HF output uses `audio` and `normalized_text`, with embedded WAV bytes and
`Audio(decode=False)` for compatibility with the inspected notebook. Include a
tested loader returning train/validation datasets and an example replacing the
notebook's source dataset cell. A converter feeds each split through the notebook's
24 kHz `processor.encode_audio` path. Preserve language and configurable voice
system prompt; Swedish export readiness does not claim the English-oriented
notebook or base model has validated Swedish fine-tuning performance.

Manifests include accepted/rejected state, provenance, source/reference hashes,
model/dependency versions, synthesis settings/seed, all QC results, manual override
history, duration, and WAV hash. Exclude endpoint credentials and other secrets
from portable configuration and logs.

## Delivery phases and verification gates

1. **Functional MVP:** GUI/project creation, reference voice selection/test,
   verified native Chatterbox adapter, responsive synthesis, atomic WAV files,
   accepted sample storage, and CSV export. Validate a real voice-to-dataset run
   when a permitted reference and model runtime are available.
2. **Automation:** weighted bilingual text, imports/local LLM, persisted jobs,
   accepted-target replenishment, bounded retries, pause/resume/stop and recovery.
   Inject crashes and test file/database reconciliation without models.
3. **Quality control:** audio QC, faster-whisper matching, automatic retry/review,
   independent calibrated speaker checks, browser/playback/waveform and edits.
4. **Training integration:** all export formats, grouped deterministic splits,
   manifests/integrity checks, HF loader and official-notebook converter tests.
5. **Optimization:** paginated large datasets, bounded queues, measured memory,
   disk estimates, throughput/ETA, operational logs, setup/startup documentation.

Tests cover normalization, duplicate detection, resampling, WER/CER, state and
manifest consistency, missing/changed references, bad audio, generation/ASR/CUDA
failures, accepted-target replenishment, pause/stop/resume, crash recovery, split
leakage, export failure atomicity, and GUI worker lifecycle. Include a 10-recording
mocked end-to-end generation/verification/export smoke test and separate opt-in real
backend tests. Mock success does not establish real synthesis or ASR quality.

Deliver README, Windows setup/start commands, dependency management/lock evidence,
example project configuration, CLI smoke command, tests, and manual validation
instructions. Record what was actually run, environment/dependency versions,
test outcomes, and unresolved limitations. A 1,000-recording acceptance run requires
1,000 accepted verified clips; attempts alone never satisfy acceptance.

## Proposed builder decisions for approval

- Standalone repository/package; no live NeuralCompanion integration changes.
- One GPU synthesis worker; CPU/int8 ASR by default due to current free VRAM.
- One active writer per project; resumable immutable job settings and attempt history.
- 24 kHz mono PCM16 default; conservative optional trim; no aggressive normalization.
- Bounded total attempts and an explicit exhausted result when a target is unmet.
- Separate Chatterbox runtime and LFM3 training dependencies.
- Native WAV/profile cloning first; service/profile discovery only against a
  documented connected implementation; no assumed Chatterbox HTTP endpoints.
- Real GPU/voice validation is reported separately from mocked automated tests;
  model downloads and a permitted reference are required for the real acceptance run.

## Independent DeepSeek review and resolved details

A DeepSeek architecture review and independent builder check were completed before
handover. The following are lead-proposed decisions resolving their findings;
they are part of the design being submitted, not prior user decisions.

### Reproducible upstream and export contract

Repository heads resolved during inspection:

- Chatterbox: `5de7a54aa4e5e2baadb0182dde554908b48b85c2`.
- Liquid cookbook: `3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232`.

Use these immutable revisions as the starting implementation contracts and verify
the pinned file contents before writing adapters. Record upstream source hashes
in the delivered compatibility notes. A later revision requires a compatibility
check, not a silent dependency update.

The HF loader returns `DatasetDict` with `train` and `validation`. Each row has
`audio` as an HF Audio feature containing `{bytes: <complete WAV file bytes>,
path: <stable exported filename>}`, plus string `normalized_text`, `text`,
`language`, and `id` columns. Decode is disabled. The WAV bytes are valid mono
PCM16 at 24 kHz; they are not raw PCM bytes. Validate snapshot hashes before
loading. Pin and test the actual `datasets` release during implementation.
Reject corrupt mappings. Keep train and validation separate rather than adopting
the notebook's hard-coded sample counts or reshuffling the combined dataset.

### Transcript authority and immutable audio

Persist three distinct values: original source text, immutable text actually sent
to synthesis, and versioned training text (initially identical to synthesis text).
An explicit user edit updates training text, retains both originals, and marks
the sample ineligible for export until transcript revalidation or an audited
manual override. ASR scoring and exports use the current training-text revision;
ASR never writes it. Audio/speaker metrics may be reused only if the audio and
reference hashes and relevant check settings are unchanged. Text edits do not
rewrite WAVs or change their hashes. Manifest history makes any text override
explicit. Regeneration always creates a new attempt UUID and new unique path.

### Model availability, dependencies, and service discovery

First launch does not download weights. Missing weights/dependencies produce
actionable setup status. Setup invocation explicitly permits package downloads,
but model weights require a separate explicit download action or user-supplied
local paths. Show model source, license, available size information, and revision;
store downloaded model identity and hashes. Normal inference uses local files
and offline model-cache settings. No text or reference audio goes to model hosts.

Current globally visible PySide6/NumPy versions are observations, not chosen lock
versions. Resolve and validate a dedicated Python 3.11 runtime, initially targeting
the upstream PyTorch 2.6.0 CUDA 12.4 build, subject to installed-driver validation.
This CUDA choice is an implementation default, not a reason to ask the user to
choose a wheel. Chatterbox's inspected Transformers pin is 5.2.0; the inspected
Liquid training notebook installs 5.20.2. Keep training in a separate environment.

No connected Chatterbox HTTP service was provided. Do not claim an operational
service connection until an actual contract can be inspected. The provider
interface exposes capabilities, health, list_profiles, resolve_profile, and
synthesize, with timeout/error reporting and cooperative cancellation boundaries.
Native file-based profiles are functional independently of remote discovery.
An unsupported service is explicitly unavailable with a reason; this limitation
must be reported rather than counted as a tested integration.

### Worker lifecycle and backpressure

Use one long-lived QObject worker on a dedicated QThread for sequential pipeline
stages and model ownership, plus a QThreadPool capped at two concurrent auxiliary
tasks for text import/generation, export, waveform preparation, and paginated
queries. Prevent concurrent project mutation/export snapshot creation while a
writer operation holds its lock. Queue at most two pending synthesis inputs;
generate additional candidates lazily. Disable duplicate start/preview requests
while the model worker is busy and show its current operation.

Use threading events for cancellation checked between stages. Keep strong worker
and thread references through completion, connect finished/quit/deleteLater in
the correct lifecycle order, and ignore close until workers have durably stopped.
Show an asynchronous stopping state and permit continued waiting if an inference
call cannot return promptly; do not destroy its objects or forcibly kill it.
Playback uses Qt Multimedia on the UI thread; file decoding, waveform decimation,
and large database queries occur in the auxiliary pool. Playback of immutable
recordings is allowed during generation.

### Recovery, calibration, and bounded targets

SQLite owns mutable project state. JSON is a versioned portable snapshot applied
only on explicit create/import/apply operations under the project writer lock.
Reject stale revision updates. Acquire locks using an OS-held file lock where
possible; operating-system process exit releases ownership. Do not reclaim a
live owner's lock merely because inference delayed a heartbeat.

A previously accepted row with missing or changed audio retains its audit status
but gains an integrity hold: exclude it from valid accepted counts/exports and
require explicit repair/regeneration. An orphan published attempt is quarantined
unless its journal proves identity, content, and matching configuration. Publish
new paths with exclusive/no-clobber semantics, not an unchecked overwrite-capable
replace. An existing destination is an error. Reconcile under the writer lock.

Budget exhaustion stops the job with `needs_attention:attempt_budget`; text
exhaustion stops with `needs_attention:needs_text`. Do not show an ETA without
usable pending/replacement text and sufficient accepted-throughput history.

Speaker calibration requires user-supplied same-speaker examples and independent
other-speaker negatives with recorded provenance. Validate class sizes and score
separation against the selected model's documented guidance; do not invent a
universal sample count or similarity threshold. Insufficient calibration leaves
the optional gate unconfigured. Manual overrides remain visibly manual.

GPU startup reports measured available memory and checks a configurable minimum;
actual peak fit remains an empirical result. Disk estimates count retained source
audio, every retained attempt, and export copies separately. Export logs and any
optional rejected files are part of the immutable snapshot and never appended
after publication. UTF-8 CSV uses standard quoting; LJSpeech export rejects text
containing pipes/newlines with actionable row errors rather than inventing escaping.

The 1,000-accepted-recording run remains a delivery acceptance gate, dependent on
a user-selected permitted reference and real model availability. No reference was
provided in the request. Implement and verify the model-free pipeline first;
report the real acceptance gate as unverified until its actual output exists.
Do not downgrade this gate to a successful mock run.

Approval of this document authorizes implementation planning followed by all five
phases, with DeepSeek agents providing scoped proposals that the lead reviews
before applying. No additional design checkpoint is proposed unless new information
materially changes the agreed scope.
