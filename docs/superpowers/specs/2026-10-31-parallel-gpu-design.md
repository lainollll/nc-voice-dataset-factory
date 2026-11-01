# Parallel GPU generation follow-up

The user explicitly requested concurrent WAV creation and GPU use. This supersedes
the original single-GPU-worker implementation constraint; existing stored jobs keep
their saved settings and datasets are not regenerated.

The chosen approach uses one or two persistent spawned generation processes, each
with its own model and RNG state. Shared-model threads were ruled out because
Chatterbox mutates voice conditioning and uses global RNGs. No batch API is assumed.
A single QC thread owns ASR/speaker checks; the existing coordinator alone writes
WAVs and SQLite. Common attempt helpers preserve the serial pipeline's QC, retry,
hash, duplicate and audit behavior. No new packages are required.

In-flight attempts are bounded to generation workers plus one QC slot. Count-only
targets reserve slots, preventing overacceptance. Duration targets can exceed the
target by bounded in-flight clips. Pause/Stop drain work; pause retains models,
Stop releases processes. OOM halts submission and reports attention required.

GPU requirement rejects CPU settings and missing CUDA. Loaded native parameters are
checked for CUDA placement. Primary cuda:0 is supported. Preflight uses conservative
free-memory measurements and an editable estimated budget plus reserve; it cannot
guarantee future allocations. CPU/int8 ASR is the default alongside GPU synthesis.

New GUI defaults: two generators, overlap enabled, GPU required. Old JSON configs
default to one generator and serial QC. Operational settings remain in immutable job
snapshots; changed worker counts apply to new jobs. Runtime GPU/worker data appears
in Batch Generator and per-attempt provenance. The in-app guide explains the controls.

PyTorch's CUDA process guidance requires spawn/forkserver, hence explicit spawn:
https://docs.pytorch.org/docs/2.6/notes/multiprocessing.html

Validation: 79 tests, actual Windows spawn CLI smoke, and ten native English clips
per mode. Sequential 93.230 s; two CUDA workers + overlap 66.010 s, 1.412x throughput.
Both runs accepted all ten; two GPU worker PIDs and five overlapping inference pairs
were recorded. Parallel export integrity passed. No 1,000-recording run was resumed.
