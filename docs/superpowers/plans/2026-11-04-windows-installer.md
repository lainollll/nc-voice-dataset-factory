# Windows Installer Implementation Plan

> **For agentic workers:** Implement sequentially in this session using the existing setup pipeline. Leave all changes uncommitted unless requested.

**Goal:** Produce a Windows online setup executable and installation instructions that a new user can follow.

**Architecture:** Inno Setup installs an explicitly staged source payload for the current user. A logged PowerShell wrapper invokes the existing dependency setup and verifies its result; no application runtime code changes.

**Tech Stack:** Inno Setup 6.7+, Windows PowerShell 5.1, existing Python 3.11/PySide6 runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-windows-installer-design.md`

## Global Constraints

- Preserve the current `.venv`, projects, datasets, model weights, and training runtime.
- Reuse `scripts/setup_windows.ps1` and its pinned dependency installation.
- Install for the current user in a writable folder; retain user-generated files on uninstall.
- Do not launch the app or report runtime readiness after dependency setup fails.
- Do not commit, push, or publish a release for this task without a request.

## Review Focus

- Paths containing spaces: compilation, dependency setup, and shortcuts must preserve arguments.
- Missing or failing prerequisites: preserve logs and return a nonzero result.
- Distribution contents: omit environments, datasets, caches, private files, and generated outputs.
- Reinstallation and uninstall: preserve user-generated recordings and configurations.
- Validation limits: distinguish stubbed installer tests from a clean Windows dependency download.

### Task 1: Logged runtime installation

**Files:** `scripts/install_runtime.ps1`, `tests/test_installer.py`.

- [x] Write behavioral tests with a stub of the existing setup script in a folder with spaces. Assert successful output/logging, propagation of nonzero exits, and caught exceptions.
- [x] Run the tests and observe the missing-wrapper failures.
- [x] Implement a transcript wrapper that invokes the existing setup, verifies core imports, and reports failures.
- [x] Run the wrapper tests and existing setup tests.

### Task 2: Installer and repeatable build

**Files:** `installer/NCVoiceDatasetFactory.iss`, `installer/INSTALL.txt`, `installer/Repair_Setup.cmd`, `scripts/build_installer.ps1`, `tests/test_installer.py`.

- [x] Add behavioral build tests using a stub compiler, asserting payload exclusions, checksum creation, and compiler failure propagation. Observe failures before implementing.
- [x] Stage an explicit payload allowlist and compile a versioned `.exe` with a SHA-256 checksum. Locate an installed compiler or accept an explicit compiler path.
- [x] Add the per-user wizard, shortcuts, dependency setup invocation, failure status, and success-only launch action.
- [x] Obtain the official Inno Setup compiler in portable mode inside the task's artifact folder; verify its signature before execution.
- [x] Compile and test the installer in isolated temporary folders with stubbed runtime setup, including failed setup and preservation of user data on uninstall.

### Task 3: Documentation and final validation

**Files:** `README.md`, `docs/INSTALLATION.md`, `docs/USER_GUIDE.md`, `docs/BUILD_INSTALLER.md`.

- [x] Make README lead with requirements, installer/source installation, first dataset, and recovery.
- [x] Preserve advanced usage in the linked user guide and remove stale repository statements.
- [x] Document build steps, troubleshooting, upgrade/uninstall behavior, and clean-machine validation limits.
- [x] Run the full test suite, lint, PowerShell parsing, and whitespace checks. Compile the final payload after documentation changes.
- [x] Request one fresh review of the final changes, address material findings, and deliver the local installer and checksum.

## Progress

- Baseline: existing setup tests passed (5/5).
- Work is on `installer/windows-setup`; no worktree or dependency environment changes are needed for this packaging-only patch.
- Targeted validation: 13 setup/installer tests passed, including real compiled installer execution with stubbed dependencies.
- Initial full suite: 153 passed; lint and PowerShell parsing passed.
- Final binary: 65 payload files, matching SHA-256, resolved local documentation links, and successful GUI startup from staged sources using the existing read-only runtime.
- Fresh review: fixed the README's omitted build-guide payload and labeled developer/helper workflows as source-only; no further material installer findings.
- Clean-machine online dependency installation remains a release-validation step; no real dependency installation or GPU generation was run for this task.
- Installer baseline full suite: 153 passed in 41.49 seconds. Subsequent UI work
  increased coverage to 180 passing tests; the user authorized committing, pushing,
  and publishing the updated installer as v0.1.0.
