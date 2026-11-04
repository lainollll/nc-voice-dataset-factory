# Beginner training interface

Status: guided page approved by the user; Chatterbox preview controls added to the scope.

## Goal

Help a new user train from an existing NC dataset without needing to understand
environment paths, notebook internals, or LoRA terminology before starting.

## Evidence

Fresh captures of the current application are saved under
`.artifacts/ui-audit-20261104/`. They show the Getting Started and LFM3 Fine-tuning
pages at 1250 x 850 with no project open. The training page enables Start without
a selected dataset, shows setup instructions despite an existing environment
marker, and leaves the guide on the unrelated WAV-selection step. A large blank
log occupies much of the page. No training or hardware validation was performed
for the audit.

## Proposed approach

Keep one training page, the existing dark theme, and the existing training
controller. Organize the page into three numbered sections:

1. **Choose recordings.** Explicitly select the current project or an existing
   dataset export. Show the source, voice/language when available, accepted count,
   and duration. If nothing is selected, explain how to open a project or choose
   an export. Inspect data on workers, not on the UI thread.
2. **Prepare training.** Show whether the separate training environment is ready,
   missing, or busy. Keep its existing setup/repair actions. Show the next valid
   action and explain unavailable actions before the user clicks them. Never
   describe an environment marker or generation GPU check as proof that a real
   training run will fit or succeed.
3. **Train and find saved files.** Preserve the existing 200-step default and
   explain it as a short initial run, not a quality guarantee. Show the real
   loading/encoding/training/saving stages from backend events, with numerical
   progress only where the backend supplies it. Show the saved adapter location
   and a clear action to open the files after a successful run.

Keep optional voice overrides and custom settings under Advanced. Keep the log
available under Technical details. Retain Stop and save; make Force stop a
secondary action with an explicit confirmation of its existing data-loss risk.
Make the guide show training help when entering training.

## Alternatives

- **Next/Back wizard:** clearer separation, but extra navigation and additional
  UI state. Prefer this only if the user wants a wizard.
- **Copy/layout cleanup only:** smaller change, but leaves dataset selection and
  readiness largely implicit.

The user approved the guided page and requested easy Chatterbox preview controls.

## Chatterbox preview

Reorganize Voice Manager using its existing widgets and settings bindings. Keep
reference/model/language selection, temperature, exaggeration and CFG controls
easy to reach. Provide editable preview text, Generate preview, Replay, Stop
playback and Play reference actions. Move less common synthesis controls and
documented-service controls into expandable sections without removing them.

Reuse `create_tts`, the existing worker runner, WAV publication, and media player.
Read settings once on the UI thread and capture the output folder before starting
work. Permit a preview without creating a dataset project, using an app-local
previews folder; never add a preview to accepted dataset records. Compute and
validate a newly confirmed reference hash on the worker without silently saving
the voice or replacing an existing saved reference hash.

Keep preview settings and saved settings bound to the same existing controls.
Provide a shortcut from training and Getting Started to this preview, carrying
the selected WAV when explicitly requested. Explain that Chatterbox previews do
not preview or activate a trained LFM3 adapter. Keep default synthesis values and
the existing 200 training steps unchanged.

## Boundaries and validation

Reuse existing dataset validation, project statistics, environment setup, and
TrainingController signal/slot paths. Preserve generation/training exclusion,
saved settings, output locations, current stop semantics, and all pipelines.
Do not install packages, download models, alter existing datasets, or run real
training to validate the UI. No checkpoint-resume or automatic NeuralCompanion
activation is added or implied by this change.

Add behavioral UI tests for missing data, readiness, source switching, busy and
failure states, progress events, saved output, and force-stop cancellation.
Render the updated screen at normal and narrow sizes; verify keyboard focus,
readable labels, unclipped controls, and that advanced controls remain reachable.
Rebuild the installer after changes are verified, without committing or publishing
unless requested.
