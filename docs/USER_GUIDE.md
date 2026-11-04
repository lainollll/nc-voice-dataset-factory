# Advanced user guide

Install the app first using [the installation guide](INSTALLATION.md).
Run commands below from the application folder.

## Create a real dataset

After setup, double-click **Start_NC_Voice_Dataset_Factory.cmd** in the application
folder to open the app. It uses this folder's `.venv`, works from other working
directories, and keeps startup errors visible. Missing setup displays instructions.
You can also pass `-Project "C:\Projects\MyVoice"` to the launcher.

1. Dashboard: create/open a dedicated project folder.
2. Voice Manager: select a WAV, enter a name, choose English or Swedish and confirm
   voice-use permission. Select a local model folder and save the project/profile.
   The saved reference hash is checked on every synthesis call; changed or missing
   references stop generation. Use Generate voice preview and listen before a large batch.
3. Text Generator: generate a weighted bilingual corpus, import TXT/CSV/JSONL/JSON,
   or configure an OpenAI-compatible local LLM base URL such as
   `http://127.0.0.1:1234/v1`. The editable model dropdown automatically fills from
   the server's `/models` endpoint when you use the LLM source. Use **Refresh models**
   to update/retry, or type a custom model ID. Saved/custom IDs are preserved.
   Ollama can use its OpenAI-compatible `/v1` interface. No local server is assumed
   to exist. Review/edit the spoken-text preview on Batch Generator.
   The editor shows the first 500 sentences; the full generated/imported corpus is
   retained for the job. Repeated LLM responses have a bounded retry budget.
4. Quality Control: select a complete **faster-whisper / CTranslate2** model folder,
   not an OpenAI Whisper `.pt` checkpoint. CPU/int8 is the default. ASR verification
   is enabled by default; missing weights prevent unchecked acceptance. Configure
   WER/CER (default 0.10), duration (3–15 seconds), silence/clipping and signal limits.
   Comparison handles clear English thousands separators and common unit spellings;
   ambiguous decimal/date readings remain visible for review.
5. Batch Generator: set accepted count (for example **1,000**) and/or accepted
   seconds, retries and a total attempt budget. When both targets are set, both
   must be met. Start; monitor accepted results, actual throughput, ETA and errors.
6. Dataset Browser: filter/search, play audio, inspect waveforms, source/ASR text,
   WER/CER, speaker metrics and synthesis settings. Select several rows for batch
   accept/reject/retry/soft-delete. Reviews require a reason and record the OS user.
   Random accepted review offers up to ten clips to listen to before export.
7. Export: choose a **new** destination folder, formats, deterministic split seed
   and validation fraction. Integrity checks must pass before publication.

Stopping and pausing finish the current sample to a durable boundary. Resume uses
the job's saved model/voice/quality settings rather than silently adopting current
controls. Restarted projects reconcile interrupted attempts and flag missing or
changed accepted audio. Failed/rejected attempts and original audio remain available.
Retries always get a new seed and filename. Accepted WAVs are never overwritten.
Delete is an audited soft delete; source reference files are never deleted.
Regenerate synthesizes the current training text into a new attempt and preserves
the original text/audio history. Retry retains the current synthesis text. A stopped
job's attempt budget can be explicitly extended with an audited reason.

An exhausted corpus, attempt budget, low disk space or CUDA OOM is **not success**.
The job reports why it needs attention. Add text or resolve the reported condition,
then explicitly retry/resume. Rejections do not count toward the accepted target.
The built-in corpus is finite: restrictive category/length combinations can exhaust
it. Imports and local LLM generation support larger and more varied text sets.
Lexical/category/character coverage statistics are reported; character coverage is
explicitly a phonetic proxy, not a claim of full phoneme coverage.

Text Generator's **Built-in corpus** combines bundled English/Swedish sentence
templates with names, places, objects, numbers, dates and other word choices.
It runs offline and does not require a text model or downloaded book corpus.
Category weights control relative proportions (20 is twice 10; zero disables a
category), not percentages that must sum to 100. Custom topics affect templates
that contain a topic slot. Very narrow word limits or category choices reduce
the available variety.

The **OpenAI-compatible LLM** source asks your running text model for new sentences,
then normalizes, deduplicates and filters them by word count. Discovery runs in a
background worker, respects remote-endpoint permission, and discards replies from
previous URL/permission selections. HTTP errors leave custom model entry available.
Servers requiring API-key authentication remain outside the existing unauthenticated
local-server setup. Choose a chat/instruct model; the server list can also contain
models intended for other tasks.

**Import a text file** accepts TXT lines, CSV with a `text` column, and JSON/JSONL
strings or objects with `text`. Imports retain their supplied content through the
existing normalization and invalid/duplicate-row reporting; preview count, word
limits, topics and category weights do not filter them. The page hides settings
for other sources and preserves their values for when you switch back.

**Preview sentence count** prepares text now. The accepted recording/minute target
belongs to Batch Generator. Review/edit prepared sentences there before starting;
for built-in/LLM jobs the existing lazy source can supply more text as needed.

## Parallel generation and GPU use

Batch Generator now offers **1–8 generation workers**, **quality-check overlap**,
and **Require GPU**. New GUI/quick setups enable **Automatically choose GPU workers**,
with an upper limit of eight, overlap enabled and native CUDA required. Auto chooses
the largest count fitting current free VRAM, the per-worker estimate and reserve
when starting a new job or resuming a stopped job. Uncheck Auto for an exact manual
count. Previously saved jobs without this option retain manual behavior. Pause and
Resume within an active run keep their existing worker pool.

Select `cuda` in Voice Manager. CPU fallback is disabled when
GPU is required; the loaded speech model's parameter placement is also checked.

Each generation worker owns its own Chatterbox model and random-number state in a
spawned process. The next clips can be synthesized while QC workers check earlier
WAVs. Auto selects up to four concurrent CPU transcript checks from CPU/RAM budgets,
sharing one faster-whisper model through its native worker pool. Manual mode, GPU
ASR and speaker verification retain one checker. ASR defaults to CPU/int8 to preserve GPU memory. All file/SQLite
updates remain on the single coordinator; accepted counts, retry budgets, unique
filenames and every enabled QC gate still apply.

The VRAM check uses the smaller of CUDA and `nvidia-smi` free memory, an editable
per-worker estimate (default 4 GiB), and at least 1 GiB reserve. This estimate is not
a memory limit or a guarantee: larger models/longer clips can need more. If memory
is insufficient, free VRAM or choose one worker; overlap still accelerates that mode.
If you put ASR on CUDA, increase headroom for it as well. GPU OOM stops new submissions
and reports NEEDS_ATTENTION; existing in-flight work is drained safely.

Auto additionally reserves 4 GiB for enabled GPU transcript checking; CPU/int8 ASR
remains the default. Runtime status and quick-setup progress show the chosen worker
count plus active generation/checking tasks and queued checks, while the saved job
retains its upper limit. Active tasks include startup and processing; they are not
a hardware-utilization measurement. Pool size and bounded work queues
use the chosen counts. Auto sizes resource-based concurrency; it does not continuously
measure utilization or benchmark the fastest worker count. More available workers
can help throughput, but model size and CPU quality-check speed still matter.
The CPU/RAM budget is an estimate, not a hard memory limit.

Pause/Stop finish up to the bounded in-flight window (GPU workers + QC workers).
Paused workers retain their models for quick Resume; Stop releases the workers.
Count-only targets reserve acceptance slots so they remain exact. Duration targets
may exceed the requested duration by the in-flight clips.

Existing saved jobs keep their original worker/device settings. To change these,
configure the controls and **Start a new job**. Native multi-worker mode currently
targets `cuda:0`; documented HTTP services use one worker because their GPU ownership
is outside this app's control.

On the tested RTX 4090, the same 10 English source sentences all passed native ASR/QC:
sequential **93.2 s**, two CUDA workers plus overlap **66.0 s** (**1.41× throughput**,
including model startup). Two distinct worker processes had overlapping CUDA
inference intervals. Results depend on model, text, VRAM and CPU/ASR load.

## Existing models and explicit downloads

Use existing local weights whenever possible. Voice Manager's **Inspect model
download** action retrieves the selected model's source, license, exact revision and
available size information. **Download inspected model** is a separate explicit
action and requires an empty destination. There are no automatic first-launch or
first-generation downloads, and voice/text material is not sent to model hosts.

The native adapter uses the installed library's verified `from_local` interface.
For multilingual models it selects V2/V3 explicitly and contains the upstream
tokenizer's Cangjie fetch so it can only resolve the selected local mapping file.
It does not edit the Chatterbox installation or disable its PerTh watermark.

English folder: `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`,
`tokenizer.json` (plus optional `conds.pt`). Multilingual: `ve.pt`, `s3gen.pt`,
`t3_mtl23ls_v2.safetensors` or `t3_mtl23ls_v3.safetensors`,
`grapheme_mtl_merged_expanded_v1.json`, `Cangjie5_TC.json` (optional `conds.pt`).
ASR requires `model.bin`, `config.json`, `tokenizer.json` and its accompanying model files.

Explicit CLI download example:

```powershell
.\.venv\Scripts\python.exe -m nc_factory --download-model multilingual-v3 --destination H:\Models\ChatterboxV3
```

Model source/revision/file hashes are saved in `nc_model_identity.json`.

## Optional independent speaker verification

Use the complete local `speechbrain/spkrec-ecapa-voxceleb` model, including
`hyperparams.yaml`, embedding/normalization/classifier checkpoints and label encoder.
The adapter loads local Pretrainer files directly and never calls its network fetch
path. Speaker embedding is independent of Chatterbox's voice encoder.

Calibrate with several distinct same-speaker recordings and independent other-speaker
recordings you have permission to use. Calibration excludes the reference itself,
records file hashes, requires separated score classes, and derives a threshold from
those labeled scores. This is empirical calibration, not a universal identity
threshold or a statistical accuracy guarantee. Validate with held-out examples.
An enabled but missing/uncalibrated/failing speaker check prevents automatic
acceptance. Reference/model changes invalidate the calibration. Manual overrides
remain visible in the manifest.

## Documented local services and voice providers

Native Chatterbox has no assumed HTTP API or saved-voice registry. The app supplies
local named WAV profiles and a `TTSBackend` protocol for future NeuralCompanion or
other-engine integration. A documented local service can use the **Documented local
HTTP service** engine and a mapping JSON referencing its actual OpenAPI 3 document:

```json
{
  "openapi_path": "your-service-openapi.json",
  "base_url": "http://127.0.0.1:YOUR_PORT",
  "operation_id": "THE_ACTUAL_DOCUMENTED_POST_OPERATION_ID",
  "request_body": {
    "ACTUAL_TEXT_PROPERTY": "{text}",
    "ACTUAL_REFERENCE_PROPERTY": "{reference_wav_base64}"
  }
}
```

The names above are placeholders, **not an endpoint specification**. The adapter
finds the operation's documented path, checks required/allowed JSON properties,
accepts documented WAV or explicitly mapped base64 JSON responses, and restricts
connections to loopback without redirects/proxies. Other placeholders include
`{language}`, `{voice_id}`, `{seed}` and supported synthesis parameter names.
An optional documented GET operation can expose profile objects. Add its actual
`voices_operation_id` and a `provider_fields` object mapping `id`, `name`, and
optionally `language`/`reference_path` to the declared response field names. The
response must be an array of objects; language values must be `en` or `sv` and
reference paths must be absolute local WAV paths. Voice Manager's explicit Fetch
action lists those profiles on a worker; selecting one preserves its stable service
ID and requires voice-use permission confirmation. A reference WAV remains required
for hashing and optional independent speaker QC. Multipart/streaming,
arbitrary remote services and undocumented routes are not claimed as supported.

## Export and Liquid LFM3 training

Every export has `audio/`, `metadata.csv`, `train.jsonl`, `validation.jsonl`,
`manifest.jsonl`, `dataset_info.json`, `project_config.json`, `rejected/`, and `logs/`.
Selected formats add `metadata_ljspeech.csv`, a standalone `load_lfm3_dataset.py`,
and/or a saved Hugging Face `DatasetDict`. WAVs are **24 kHz mono PCM16**. Optional
working originals are float32 WAVs at the actual synthesis rate. No aggressive
loudness normalization or intentional watermark removal is performed.

Exports default to accepted-only. The explicit rejected-audio option copies those
files separately and never places them into training splits. Exact audio/text
duplicates, changed hashes, invalid audio, stale transcript verification, bad
LJSpeech delimiters, missing files and mapping/split errors block publication.
Exact/near transcripts are grouped before deterministic splitting; a connected
group never crosses train/validation. Whole groups can make the achieved validation
fraction approximate. A one-group dataset remains train-only with a warning.

The [official Liquid notebook](https://github.com/Liquid4All/cookbook/blob/3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232/finetuning/notebooks/lfm3_audio_tts_sft_with_transformers.ipynb)
expects WAV bytes under `audio` with decode disabled, and `normalized_text`.
Use the export's standalone loader in the **separate notebook training environment**:

```python
from load_lfm3_dataset import load_lfm3_dataset, encode_for_lfm3
raw_ds = load_lfm3_dataset(r"H:\Datasets\NC_Voice_Dataset")
encoded_ds = encode_for_lfm3(raw_ds, processor)  # official notebook's loaded processor
train_dataset = encoded_ds["train"]
eval_dataset = encoded_ds["validation"]
```

The loader validates mappings and hashes before reading audio and streams WAV bytes
into a disk-backed Arrow cache rather than keeping the whole dataset in RAM.
Its optional `cache_dir` selects that working cache location.

Exports contain recordings for preprocessing. Fine-tuning can now be launched
from the **LFM3 Fine-tuning** card or **Fine-tune with LFM3** on Getting Started:

1. In **Choose your recordings**, select **Current project** or **Saved dataset
   export**. Check the voice, language, accepted count, and duration shown below.
   Missing/empty selections and mixed-voice data prevent Start. Saved exports are
   integrity-checked when selected; current-project audio is rechecked when the
   training snapshot is prepared.
2. In **Prepare training**, click **Set up training environment** if needed. It
   creates `lfm3-training/env`, separate from the speech-generation `.venv`, and
   installs training dependencies. **Ready** means the setup marker exists; it
   does not guarantee GPU compatibility or sufficient GPU memory for training.
3. In **Train and save your voice**, click **Start fine-tuning**. Voice name is
   read from the dataset; the existing default is 200 steps, a batch size of one,
   and gradient accumulation. Optional voice-name overrides and custom step counts
   remain under **Advanced training options**. The app fetches the pinned notebook,
   replaces its LJSpeech/demo cells with your saved splits and voice prompt, and
   runs model loading, codec encoding, LoRA setup and training outside the UI process.
4. Use **Stop and save** for a cooperative stop at a training-step boundary. During
   model loading, stop waits for that stage. **Force stop** terminates the owned
   process tree after confirmation; current unsaved work or an in-flight checkpoint
   may be lost. The log and emergency control are under **Technical details and logs**.
5. **Open saved voice files** opens the saved adapter after a successful run.
   **Open run folder / checkpoints** under Technical details opens
   `lfm3-training/runs/<run-id>`. It contains the
   adapted `finetuning/notebooks/lfm3_audio_tts_sft_with_transformers.ipynb`,
   `training.log`, checkpoints, and the trained LoRA adapter plus processor in
   `artifacts/adapter`. It does not automatically merge or activate the adapter.

Speech generation and fine-tuning cannot run concurrently. The initial training
run downloads LFM3 weights into the separate training cache. Setup uses CUDA 12.8
PyTorch/torchaudio 2.9.1, Transformers 5.20.2 and datasets 3.6.0. The retained training
cells do not use TorchAO or the notebook's optional SpeechBrain speaker evaluation;
those packages are omitted (TorchAO 0.18 has no Windows wheel). This is an adapted
training path, not a claim that every original demonstration cell runs on Windows.

Training setup pins NumPy 1.26.4 and PyArrow 20.0.0 together. An unpinned Arrow
upgrade could otherwise require NumPy 2 and break dataset imports. If an earlier
setup failed with that error, restart the app and run **Set up training environment**
again; it repairs the existing training environment. Setup verifies actual Arrow
array and dataset operations before marking the environment ready.

Do not mix this environment with Chatterbox's Transformers 5.2.0 runtime. The notebook
requires a CUDA GPU with bfloat16 support; actual memory needs depend on recordings
and model runtime. Swedish export uses the same schema, but Swedish LFM3 training
quality remains unverified. No real LFM3 training was run during implementation.

Immutable synthesis text, original source text and versioned training text are
separate. An edit invalidates transcript verification until revalidation or an
audited manual override; ASR never edits the source. Export uses the current
validated training text and retains synthesis text and review history in provenance.

The default export matches [MikhailT/lj-speech](https://huggingface.co/datasets/MikhailT/lj-speech):
**LJ Speech-style Hugging Face Parquet (full split, 22,050 Hz)** is already selected
when the app opens and is also the backend/CLI default. Other formats are optional.
This default needs the export dependencies installed by `scripts/setup.ps1 -Exports`.
The
`ljspeech_hf/` subfolder contains a dataset card and `data/full-*.parquet` shards
with exactly `file`, `spoken_text`, `normalized_text`, and `audio` columns. WAV
audio is embedded, resampled to 22,050 Hz mono PCM16, and checked against a hashed
source mapping. `spoken_text` retains the actual synthesis text; `normalized_text`
is the current validated training transcript. All accepted examples form `full`.
The reference corpus and its license are not copied into your generated dataset.

```python
from datasets import Audio, load_dataset
full = load_dataset(r"C:\Datasets\NC_Voice_Dataset\ljspeech_hf", split="full")
full = full.cast_column("audio", Audio(decode=False))  # WAV bytes for preprocessing
```

The parent export retains its existing 24 kHz audio, manifests and deterministic
train/validation subsets for LFM3. The compatible Parquet output is a separate
format; it uses bounded 16-record writes and an estimated 256 MiB raw-audio shard
budget. The 817-recording real export was reloaded and verified in this format.

## Tests and diagnostics

The test and lint commands below require a **source checkout** with development
dependencies installed using `scripts/setup.ps1 -Development`. The Windows
installer does not include the tests, examples, or development tools.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts examples
```

These application diagnostics and validation commands are also available in an
installer installation. Run them from the application folder:

```powershell
.\.venv\Scripts\python.exe -m nc_factory --diagnostics
.\.venv\Scripts\python.exe -m nc_factory --smoke .\new-smoke-output --language sv --count 10
.\.venv\Scripts\python.exe -m nc_factory --validate H:\Datasets\NC_Voice_Dataset
```

Smoke output contains explicitly labeled synthetic tones and source-backed mock
ASR. It verifies pipeline/QC/scoring/export plumbing and **does not establish speech
quality, cloning fidelity, or recognition accuracy**. Automated tests also inject
missing references, bad/non-finite audio, generation failures, CUDA OOM, crashes,
hash drift, pause/restart, edit invalidation, export corruption and worker shutdown.

From a source checkout, `scripts/prepare_validation.py --help` explains how to
prepare a real run from existing local files; permission confirmation is required.
The helper is not included in the Windows installer. Once you have a configuration
file, either installation supports:

```powershell
.\.venv\Scripts\python.exe -m nc_factory --run-config .\my-config.json --project H:\Projects\MyVoice --export-to H:\Datasets\MyVoice
```

The CLI prints its persistent job ID. Control a running CLI job from another shell:

```powershell
.\.venv\Scripts\python.exe -m nc_factory --control pause --project H:\Projects\MyVoice --job-id JOB_ID
.\.venv\Scripts\python.exe -m nc_factory --control resume --project H:\Projects\MyVoice --job-id JOB_ID
.\.venv\Scripts\python.exe -m nc_factory --control stop --project H:\Projects\MyVoice --job-id JOB_ID
.\.venv\Scripts\python.exe -m nc_factory --resume-job JOB_ID --project H:\Projects\MyVoice --export-to H:\Datasets\MyVoice
```

Use one active writer per project. GUI close and CLI controls never forcibly kill
inference threads. Operational logs rotate under each project's `logs/`; SQLite
stores audit events and every attempt. Project and export directories are isolated.
