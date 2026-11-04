# Windows installer and installation guide

Status: online installation selected by the user on 2026-11-04.

## Intent

Provide a downloadable Windows `.exe` installer and clear instructions for people
installing NC Voice Dataset Factory from its GitHub repository.

## Recommended approach

Build a small Inno Setup installer containing the application source and the
existing setup scripts. Install for the current user into a writable application
folder, with Start menu shortcuts to launch the app, repair dependency setup, and
read the installation guide. Offer an optional desktop shortcut.

Run the existing `scripts/setup_windows.ps1` process during installation. It finds
or installs 64-bit Python 3.11, Git, and FFmpeg, then installs the pinned runtime
into that installation's `.venv`. Show installation progress and preserve an
installation log. A failed runtime setup must be reported as a failure, with
clear retry instructions; it must not claim that the app is ready.

This is an online installer: dependencies require internet access. NVIDIA drivers
remain a documented prerequisite for GPU generation. Speech and transcription
model weights remain explicit downloads through the app's existing model guide.
The separate optional LFM3 training environment keeps its existing setup flow.

## Alternatives considered

- An offline bundle would include several gigabytes of dependencies and require
  additional redistribution and platform validation. It is outside this proposed
  first installer; use it instead if offline installation is a user requirement.
- A frozen executable would change dependency loading, application-root paths,
  multiprocessing, and training subprocess behavior. Packaging the existing
  supported Python environment is the smaller and safer change.

The user selected downloading Python and required packages during setup.

## Files and behavior

- Add an installer definition under `installer/` and a repeatable build command
  under `scripts/`. Copy from an explicit allowlist of application files.
- Produce a versioned setup `.exe` and SHA-256 checksum under `dist/`.
- Replace the README's development-first presentation with a concise installation
  and first-run guide. Keep detailed usage in a linked document.
- Include troubleshooting for missing WinGet, interrupted downloads, dependency
  errors, missing models, GPU requirements, and resuming a saved project.
- Include manual source installation as an alternative when no release asset is
  available. Do not claim an unpublished GitHub release exists.
- Preserve user projects, recordings, exports, model weights, and optional training
  data during upgrades and uninstall. Do not package existing local environments,
  datasets, caches, secrets, or build outputs.
- Leave synthesis, quality checks, saved settings, project recovery, and all app
  code behavior unchanged. Do not run setup against the current working app's
  `.venv` or start a real generation/training job during validation.

## Validation

Compile the installer with Inno Setup and verify its file inventory. Exercise
installation and failure handling in separate temporary folders using stubbed
prerequisites to avoid installing or changing system dependencies. Verify paths
with spaces, failure reporting, shortcuts, repair, and preservation of user data.
Run the relevant existing setup tests and document the limits of clean-machine
validation. A full real online dependency install requires a clean Windows test
environment; do not represent a stubbed installation as that test.

## Delivery

Deliver the local installer, checksum, updated README, installation guide, and
build instructions. Leave changes uncommitted and unpublished unless the user
requests committing, pushing, or creating a GitHub release for this work.
