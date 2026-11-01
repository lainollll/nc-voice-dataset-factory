No confirmed load‑bearing defects found in the reviewed paths.

I checked specifically for the failure classes you listed and could not construct a concrete symptom triggered by this code:

- **Deadlock / lost attempts:** all `generating`/`checking` futures are drained before the `not outstanding` branch runs, `continue`/`wait(..., 0.05)` guarantee liveness even with `overlap_quality=False`, and `finally` shutdowns use `cancel_futures=False` so allowed in-flight work is journaled on the coordinator thread rather than abandoned.
- **Exact count reservation:** the guard `stats["accepted"] + outstanding >= target_count` combined with `outstanding` counting generating+checking+ready matches every path where an attempt can later become accepted; `_fail_work` moving a sample back to PENDING is only reached after the attempt is already counted in `attempts`, so no path can accept beyond target on the count branch.
- **Typed GPU placement:** `gpu_preflight` rejects `cuda:1+`, `require_cuda` on cpu engines is rejected upstream (per test), and the process pool receives `cuda:0` only. Injected-TTS path is pinned to 1 worker, so no mixed-device workers.
- **Coordinator unexpected exception:** the `try/finally` around executors ensures `generation`/`quality` are shut down even on an unexpected exception in `_save_generated`/`_begin_work`. The outer `Pipeline.run` owns the writer lock/status transition (per your stated design), so the coordinator-thread lock is released by the caller, not silently stranded here.
- **Data corruption / concurrent writes:** all DB and file writes occur on the coordinator thread; workers return `AudioResult` (numpy) or `QualityResult` only; ASR/speaker QC is confined to one `audio-qc` thread by `max_workers=1` + single `QualityWorker` instance, satisfying the "1 QC thread owns ASR" invariant.
- **`BrokenProcessPool` / OOM:** caught via `future.result()`, both map to `fatal`/`outcome == "oom"`, and the `shutdown(wait=True)` drains remaining futures so no submission is orphaned while status is set to `NEEDS_ATTENTION`.
- **Retry/global budget:** `exhausted` is reset to `False` on each successful submit and recomputed when `next_pending()` returns `None`; `attempt_budget` is enforced both at fill time and at the idle checkpoint, so overshoot requires only the bounded in-flight window you already documented.

There are minor robustness/efficiency notes (not defects under your stated semantics) that I am deliberately not proposing fixes for, since they don't meet the bar:

1. If `_save_generated` raises after `finish_attempt(GENERATED)` but before `update_sample(VALIDATING)`, the sample could be stuck in `GENERATED` — but you've said pause/stop drain tests assert no `GENERATED` states remain, so this is either unreachable or covered by `_fail_work` ordering in the real `_fail_work`; not confirmed from this excerpt.
2. `pipeline.text_source` is called from the coordinator thread inside the fill loop; if it ever did I/O on the written repo that would be safe (coordinator owns writes) but blocking on external input could delay `_external_control` — not a defect, and fixing it would risk the asyncio/threading constraint.
3. Preflight uses `min(torch_free, nvidia_smi_free)`; if `nvidia_smi` reports free only of process‑0 device and `torch` did not report, `free=0` blocks ALL parallel runs — but your test exercises this exact path, so this is intentional fail‑safe.

**Suggested validation steps (no changes ran):**
- Run `pytest -k "parallel and (stop|pause|oom|retry|target)"` under `spawn`.
- Add a fault‑injection test wrapping `_save_generated` to raise once on the coordinator after a successful generation; assert `NEEDS_ATTENTION`, budget accounted, lock released.
- Add a test with `overlap_quality=False` and `generation_workers=2` asserting `len(generating) <= 1` while `checking` is non‑empty (matches drain rule).
- Add a preflight test where `torch_mem_get_info` is missing but `nvidia_smi` reports sufficient free; confirm pass (guards against the `free=0` concern above).

If you can share `_checkpoint`, `_external_control`, `targets_met`, and `next_pending`, I'd narrow further — those are the only call sites where a confirmed defect could currently hide.
