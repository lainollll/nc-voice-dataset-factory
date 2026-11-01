# Remove the measured transcript-check bottleneck

Read-only live audit found four real generation worker PIDs but about 0.92 active
inference calls on average. Recent median synthesis was 2.98 seconds, ASR 3.43 seconds,
and generation-to-finalization lag 14.3 seconds. Throughput was about 16.4 accepted
clips/minute. One CPU quality worker and a five-item outstanding-work limit kept
most generators idle. This is observational evidence, not a controlled 2-vs-4 test.

Auto will select CPU quality concurrency from logical CPU and available RAM budgets,
capped at four. Keep single checking for manual mode, GPU ASR, speaker verification,
and injected checker objects. Reuse the existing QualityController per thread, sharing
one thread-safe FasterWhisperASR configured with matching CTranslate2 workers. Model
weights are shared on one device. Preserve beam size, thresholds and all acceptance,
retry, duplicate, writer, Pause/Stop, target and audit behavior.

Use the resolved quality count in executor capacity, submissions and backpressure.
Show active generation tasks, active quality tasks and queued checks in runtime UI.
Record quality timing/thread diagnostics. No dependency or running-job modification.

- [x] Failing tests for CPU/RAM planning, single-load concurrent ASR, concurrent
  independent checkers and bounded exact-target scheduling.
- [x] Implement RAM probe, runtime policy, ASR load lock/worker count and thread-local
  quality checkers using the existing pipeline.
- [x] Add active/queued status, regression coverage, full tests and review.
- [x] If resource budgets permit, benchmark transcript checking on a few existing
  WAV files in an isolated CPU process; never restart or change the current job.

Verification: 143 tests passed; Ruff clean. Independent read-only review found no
actionable issues. A prior native Qt crash in a quick-setup test did not reproduce
in isolation, the complete 41-test targeted sequence, or the full suite; no cause
has been established. Runtime labels were rendered and inspected at 1200x850.

Read-only CPU benchmark: eight existing accepted WAVs, warm shared medium/int8 model,
four CPU threads per checker, unchanged beam 5/VAD. Serial passes: 23.424/23.101 s;
two-checker passes: 15.885/15.524 s, or 1.481x throughput, identical transcripts.
The live job was left running. This measures only ASR, not overall dataset speed.
Report: `.artifacts/cpu_quality_benchmark.json`.

User requested more than two checkers. Auto already supports four, with CPU/RAM
limits; added explicit four-checker cap and low-CPU assertions (six cases passed).
Three-checker real ASR comparison: serial 22.571/23.295 s versus 13.103/14.101 s,
1.686x throughput and identical transcripts. Report:
`.artifacts/cpu_quality_benchmark_3.json`. Full-job throughput remains unmeasured.
