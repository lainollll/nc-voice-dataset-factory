# LFM3 notebook training controls

User selected in-app fine-tuning controls. Add an LFM3 Fine-tuning card using the
pinned official Liquid notebook already referenced by the exporter. Keep generation
and training environments separate. The card can install a dedicated training
environment, snapshot current accepted recordings or use an existing NC export,
start training, request a cooperative stop, force-stop if necessary, and open run
artifacts. No training, environment installation or model download during development.

Use existing ExportOptions/export_dataset, standalone LFM3 loader and TaskRunner.
Use QProcess for external setup/training with bounded UI logs. Block concurrent
generation and training and wait for owned work on shutdown. Verify pinned notebook
bytes before adapting its model loading, collator and LoRA cells. Replace LJSpeech
data/demo cells with the user's validated train/validation splits. Configure the
voice prompt, small batches and step limit, save checkpoints and final adapters.
Treat failures and stops distinctly from successful completion.

- [x] Backend tests first: notebook checksum, pure-Python cells, preserved split
  selection, own voice prompt, checkpoints, validation and new run isolation.
- [x] Implement training preparation, notebook adapter and standalone runner.
- [x] Add separate-environment setup and asynchronous process controller.
- [x] Add card, export handoff, guide navigation, concurrency and shutdown guards.
- [x] Test process output/failure/stop and GUI handoff with fixture processes;
  never install packages or launch real GPU training in the test suite.
- [x] Run full suite/lint, inspect UI, review lifecycle, and document limits.

Completion: 116 tests passed; targeted Ruff clean. Inspected 1250×850 and 1000×700
layouts. Independent read-only review found no actionable issues. The real pinned
notebook was fetched, checksum verified and adapted code compiled. Environment
installation, model download, real GPU training and adapter quality are unverified.
TorchAO and the optional speaker-evaluation/demo cells are omitted; retained cells
do not require those packages, and TorchAO 0.18 has no Windows wheel. No commits.
