# Simple dataset setup implementation plan

**Goal:** Start a new dataset with a WAV, language, and accepted recording count or minutes.

**Design:** Put a simple form on Getting Started. Find complete local models in saved
projects, app-local model folders, and the Hugging Face cache. Use existing synthesis
defaults, CPU/int8 transcript checking, built-in text, automatic attempt budgets,
and one/two GPU workers according to available memory. Create a unique project and
delegate generation to the existing Batch Generator pipeline. Keep its controls,
quality checks, resume behavior, and the detachable guide. Missing models use the
existing inspect/download functions with an explicit download action into a new
app-local folder. Advanced cards remain accessible. No automatic real generation
or model download during development; no commits.

**Tech stack:** Python 3.11, PySide6, existing TaskRunner and Pipeline.

## Tasks

- [x] Add and run failing backend tests: compatible model discovery, incomplete or
  stale model paths, Swedish selection, WAV/permission validation, size and retry
  budgets, GPU capacity, new project isolation, and cancellation.
- [x] Implement `nc_factory/quick_setup.py`: immutable request/model/prepared-result
  types, bounded local discovery, configuration building, and project preparation.
- [x] Add failing Qt integration tests for prepare-to-generate, background execution,
  active-job protection, cancellation, and errors visible on the simple form.
- [x] Implement `ui/quick_start.py` and integrate into Getting Started. Queue start
  until existing workers are idle; reuse project loading, preview, job start,
  progress, Stop/Pause, and export. Never read Qt widgets from worker threads.
- [x] Add missing-model inspection and explicit download using existing model APIs.
- [x] Add a short simple-setup guide step and document the new workflow.
- [x] Run the full suite and lint, inspect rendered UI, review lifecycle and existing
  behavior, and report validation and remaining limits.

## Review focus

No UI-thread model scans or GPU probes; no silent CPU/mock fallback; no disabling
ASR; no starting a stale or cancelled request; no mutation of existing projects;
no model download without the user's in-app action; no Qt reads in worker tasks.

## Completion evidence

107 tests passed; targeted Ruff checks clean. Inspected normal and 1000×700
offscreen layouts. Read-only discovery found the previously used English/medium-ASR
paths and a complete multilingual-v2 cache. No new real voice run or download was
performed. Review found and tests reproduced lost simple Resume tracking and
English-only ASR selection for Swedish; both were fixed and re-reviewed. Compact
layout clipping was also reproduced and fixed. No commits or pushes.
