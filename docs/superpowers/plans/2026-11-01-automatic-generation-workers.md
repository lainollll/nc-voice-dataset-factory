# Automatic dataset generation workers

Use the existing GPU preflight to choose a worker count at Start/Resume. Add a
persisted `auto_generation_workers` boolean, default false for old configurations.
New GUI/quick setups enable Auto with an eight-worker upper limit. Auto selects
the largest count fitting currently free VRAM, the per-worker estimate and existing
reserve. Manual jobs keep their exact requested count and existing failure behavior.
No CPU/mock fallback, quality reduction, live pool restarts or benchmark clips.

The existing scheduler must use the resolved runtime count for pool size, admission
and bounded outstanding work. Keep the saved job's requested limit unchanged and
audit the chosen count. Display Auto/limit in runtime status and quick setup progress.
Existing overlap, CPU/int8 ASR, targets, retry budget, Pause/Stop and resume mechanics
remain intact. This sizes concurrency; it does not guarantee maximum throughput or
continuously retune based on GPU utilization.

- [x] Add failing tests for old configuration compatibility, automatic count/caps,
  insufficient VRAM, GUI settings round trip and actual scheduler concurrency.
- [x] Add Auto field/validation and localized GPU preflight/scheduler changes.
- [x] Add Auto controls and status; enable it for new quick setups.
- [x] Run full suite/lint, inspect UI and review correctness; document empirical limits.

Completion: 137 tests passed, targeted Ruff checks clean, normal UI rendered and
inspected. Independent read-only review found no actionable issues. Tests prove the
resolved count controls real threaded scheduling, stopped-job Resume rechecks free
memory, and the saved limit remains intact. No real GPU generation or speed benchmark
was started. Auto also adds 4 GiB of estimated reserve when ASR is configured on CUDA.
