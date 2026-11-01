## Verdict

**Spec: mostly met.** Core pipeline, atomic audio publication, recovery, immutable export staging, local-only model loading (Chatterbox `from_local` + patched Cangjie fetch, faster-whisper `local_files_only`), enabled-QC gating, and bounded text import/LLM request sizes are in place and consistent with the reported real run.

**Quality: block-worthy if any listed real bugs remain.** Several items below are load-bearing (worker lifetime, recovery semantics, settings save). Two are plan/limitation and excluded from must-fix count. Short list ordered by severity.

## Findings (real bugs, source-supported)

**1. `TaskRunner` worker lifetime — premature destruction / thread-affinity hazard.**
- Function: `TaskRunner.start` in `workers.py`.
- Trigger: any `run_task`/`run_aux` where the Python `Worker` loses its last reference; worker is created with no parent and only `self._worker` holds it, but `_worker` is a strong ref — so Python-side is fine. The real gap: `worker.finished.connect(thread.quit)` runs in the worker thread; `thread.finished.connect(worker.deleteLater)` after quit is the canonical pattern and is present. However `Worker.run` executes inside the worker thread via `thread.started`, which is correct. The concrete load-bearing issue: `TaskRunner.finished` is emitted inside `_on_thread_finished` slot *after* `_thread=None`; `MainWindow._finished` may call `self.close()` while other Qt objects (aux_runner, pending) still exist — but `aux` is separately guarded in `closeEvent`. Net: no lifetime bug found here; downgrade to "no issue found."
- Consequence: none; drop this item.

**2. Recovery: `accepted` states are never regenerated, but `recover()` contract is not shown; `Pipeline` relies on it to flip `GENERATING/VALIDATING` back without touching `ACCEPTED`.**
- Function: `Repository.recover` (not in provided source).
- Trigger: process crash mid-`write_wav_atomic` or mid-`update_sample`.
- Risk: if `recover()` resets `GENERATING`→`FAILED` without clearing the just-published audio file, stale audio orphaned (bounded by "one file per attempt_id"); not an overwrite risk, since `write_wav_atomic` never replaces. **Not verifiable from shown source**; flag as "verify `recover()` does not touch `ACCEPTED` and does not delete accepted audio." Call out as review gate, not a confirmed bug.

**3. `Pipeline.run` holds `repo.writer_lock()` for the entire job.**
- Function: `Pipeline.run` (open of `with self.repo.writer_lock():`).
- Trigger: any concurrent `open_project` (calls `repo.recover()` under `writer_lock`), or `export_dataset` (also takes `writer_lock`); GUI blocks both tasks.
- Consequence: with a ~1000-clip job running, opening another project or exporting will block for the job's duration. GUI's `run_task` prevents starting a second `run_task`, but `open_project` uses the same `runner` so it's mutually excluded — acceptable. **However** `closeEvent` while `runner.busy` ignores the close and calls `stop_job`; if the pipeline is between attempts and blocked, stop takes effect at next checkpoint. **Documented limitation**, not a bug. Downgrade.

**4. `repository.next_pending`/`begin_attempt`/`finish_attempt`/`update_sample` are not shown.** Load-bearing invariants:
- `accepted` count must be derived from `state='ACCEPTED'` rows only, and after revalidate must exclude `integrity_hold`.
- `statistics()['attempts']` must count rows in `attempts` table (not retries), else `max_attempts` semantics drift and current 1000-run could exhaust early.
- Flag as "verify": **not a bug claim.**

**5. `ProjectTextSource.__init__` calls `repo.statistics()` and `iter_samples()` synchronously — but it's only constructed inside the worker thread (`start_job`/`resume_job` task closure).** Safe.
- However `start_job` constructs `Pipeline(self.repo, ...)` on the **GUI thread** while `self.runner` may still be idle: `Pipeline.__init__` only stores refs — safe. **No issue.**

**6. `MainWindow.save_project` rebuilds config but drops some fields the UI does not expose.**
- Function: `configuration()` → passed to `ProjectConfig(...)`.
- Trigger: user opens an existing project that has non-default `video`/`schema_version`/future fields, saves from UI.
- Effect: `schema_version` defaults to 1 (matches), but `keep_original` is set, `target_*` are set; **all currently-defined ProjectConfig fields are wired**. Only unknown-future-field loss would occur — documented limitation given schema_version gating. **Not a bug** for schema 1.

**7. `configuration()` loses `quality.trim_margin_ms`, `check_abrupt_edges`, `check_repetition` on save.**
- Function: `MainWindow.configuration` — `QualitySettings(...)` kwargs cover `verify_asr, asr_model_path, asr_device, asr_compute_type` + `qc_spins` (max_wer, max_cer, min_duration, max_duration, max_silence_ratio, min_rms, max_clipping_ratio) + `trim_silence, speaker_enabled, speaker_model_path, speaker_calibration_path`.
- **Missing on save**: `trim_margin_ms`, `check_abrupt_edges`, `check_repetition`.
- Trigger: open project with `trim_silence=True, trim_margin_ms=200`; save quality settings page.
- Consequence: silently reverts margin to 120 ms and both trajectory checks to True on next run. **Real bug — settings save regression.**
- Small fix: include `trim_margin_ms=120` (or preserve from `self._last_config.quality`), and pass `check_abrupt_edges=self._last_config.quality.check_abrupt_edges if self._last_config else True`, `check_repetition=...` in the `QualitySettings(...)` construction — or add spinboxes.

**8. `start_job` overwrites `self.job_id` from worker task, then `_job_finished` re-reads it.**
- Function: `MainWindow.start_job` → `task()` mutates `self.job_id` from the worker thread.
- Trigger: user switches `job_selector` while `runner` is busy (it's disabled via `run_task` guard for starting but not for selection; `_select_job` is connected and writes `self.job_id`).
- Consequence: race — worker sets `self.job_id = repo.create_job(...)`, then queued `currentIndexChanged` from user click sets a different id; `_job_finished` then repairs `job_selector` from `self.job_id`, causing the *running* job to be silently deselected, and `pause_job`/`stop_job` still target `self.pipeline` (correct), but `resume_job` fallback path would pick the wrong `job_id` next time. Real but narrow.
- Small fix: have `task()` return `(job_id, result)`; set `self.job_id` in `_job_finished` handler only, or disable `job_selector` while `runner.busy`.

**9. `Pipeline._attempt` — after `finish_attempt(FAILED, ...)` sample is set `PENDING` for retry; but for a successful QC-rejected sample the loop then takes the *same* `sample` snapshot next iteration? No — `next_pending` re-reads. Fine.**
- The accepted→accepted path calls `update_sample(..., validated_revision=sample['training_revision'])`; after `review_selected('accept')` on a REJECTED row, `Repository.review('accept')` (not shown) must set `validated_revision = training_revision` and integrity_hold=False, else export refuses. **Verify** — not a bug claim.

**10. `generate_preview` — LLM loop has no bound.**
- Function: `MainWindow.generate_preview` `while len(items) < count:`.
- Trigger: local LLM returns duplicate/near-dup sentences consistently; `parse_llm_sentences` de-dupes, and `local_llm_texts` may return only a few unique.
- Consequence: unbounded loop with 90 s per request until user clicks Stop → GUI remains responsive (worker thread), but runaway traffic to endpoint. **Real bug — bounded-operation violation.**
- Small fix: cap attempts (e.g. `attempts < 6 * ceil(count/64)` then raise "local LLM did not yield enough unique sentences").

**11. `Pipeline._attempt` — for non-`oom` failure, retry reuses `sample['attempt_count']` snapshot.**
- Function: `_attempt` uses `sample["attempt_count"] < config.max_retries` from the snapshot taken before `begin_attempt`.
- Trigger: `_attempt` fails (e.g. TTS exception). Sample reset PENDING. Next loop `next_pending` reloads the row — attempt_count is now incremented (assuming `begin_attempt` increments it). **Verify** `begin_attempt` increments `attempt_count`; if not, the retry loop is unbounded for hard failures. Not confirmed.

**12. `export_dataset` uses `repo.writer_lock()` — concurrent with a running job's writer lock → serializes but does not corrupt.** Actually worse: pipeline holds writer lock for entire job duration, so export will block on `writer_lock` until job ends. In GUI, `run_task` blocks starting export while `runner.busy` — so **not reachable from GUI during a run**. **Documented limitation** (CLI `--export-to` after `run` returns). Drop.

## Documented limitations (not to fix now)

- Whole-job `writer_lock` blocks external inspection while running; GUI already serializes.
- `keep_original` FLOAT sidecars grow unbounded with retries (disk preflight partially covers).
- `NativeChatterbox.close` sets `_model=None` but not `_torch`; a subsequent backend on the same instance would reuse GPU context — acceptable since backend is per-job.
- `model_download_plan` allows arbitrary `--model-repo` — user-initiated network, matches "explicit download" UX.

## Must-do before final delivery

1. Fix #7 (settings save silently drops `trim_margin_ms`, `check_abrupt_edges`, `check_repetition`).
2. Fix #10 (bound LLM preview retry loop).
3. Fix #8 (job_id race — worker must not write `self.job_id`; return it).
4. Verify (not proposed by me): `Repository.recover` leaves `ACCEPTED` untouched and does not clear `integrity_hold`; `begin_attempt` increments `attempt_count`; `review('accept')` sets `validated_revision` and clears integrity_hold. If any fail, they become must-fix.

## Validation steps

- Add regression test: `QualitySettings` with `trim_margin_ms=200, check_abrupt_edges=False, check_repetition=False` → open project → `save_project` → reload → fields preserved.
- Add regression test: `generate_preview` with mock LLM returning duplicates → raises within bounded attempts.
- Add regression test: `start_job` while user changes `job_selector.currentIndex` — assert `_job_finished` leaves selector on the created job.
- Re-run the 46-test suite plus the accepted real run's job JSON; confirm CSV/LFM3 validation still passes.
- Confirm `recover()` unit test asserts an `ACCEPTED` row with `integrity_hold=0` and `training_revision==validated_revision` is untouched.

No code changes made; no tests run by me.
