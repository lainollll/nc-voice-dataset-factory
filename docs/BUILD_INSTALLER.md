# Build the Windows installer

The installer packages the existing Python application and runs its supported
online dependency setup. It does not freeze Python, bundle model weights, or copy
local datasets/environments into the distribution.

## Requirements and build command

Use Windows and [Inno Setup 6.7 or newer](https://jrsoftware.org/isdl.php).
The build script accepts an installed or portable compiler. It does not install
build tools automatically.

From a source checkout:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_installer.ps1
```

If the compiler is not on PATH or in a standard installation folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_installer.ps1 -CompilerPath "C:\Tools\Inno Setup 6\ISCC.exe"
```

Use `-OutputDirectory "C:\Builds\NCVoice"` to choose a different output folder.
The version comes from `pyproject.toml`.

Outputs for version 0.1.0:

```text
dist\NC-Voice-Dataset-Factory-0.1.0-Setup.exe
dist\NC-Voice-Dataset-Factory-0.1.0-Setup.exe.sha256
dist\NC-Voice-Dataset-Factory-0.1.0-Setup.exe.payload.json
```

The checksum file contains the SHA-256 hash and installer filename. The JSON
file records the staged payload inventory. The script stages only explicitly
listed application/setup/documentation files and Python sources; `.venv`,
datasets, model weights, logs, secrets, caches, and previous distributions are
excluded. Each build's staging directory is under `.artifacts/installer-build-*`.
An unsuccessful compile does not replace an existing installer output.

## Tests

Use an existing development environment; do not run the real dependency installer
against your working application just to test packaging.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_installer.py tests/test_setup_cmd.py -q -p no:cacheprovider
```

Wrapper tests invoke PowerShell in temporary folders and simulate prerequisite
success/failure. Build tests verify staging exclusions, paths with spaces, matching
checksums, and compiler failures. Compiled-installer integration tests need an
Inno Setup compiler and otherwise skip automatically. Set its location explicitly:

```powershell
$env:NC_INSTALLER_COMPILER = "C:\Tools\Inno Setup 6\ISCC.exe"
.\.venv\Scripts\python.exe -B -m pytest tests/test_installer.py -q -p no:cacheprovider
```

The compiled tests use unique test app identities/shortcut folders, stubbed runtime
setup, and temporary destinations. They verify success and failure exit status,
shortcut creation, reinstall behavior, uninstall, and preservation of user files.
They do not download or install the real runtime dependencies.

## Before publishing a release

Test the final executable on a clean Windows 11 x64 PC or VM, with a supported
NVIDIA driver where real speech generation will be exercised:

1. Download and verify the installer/checksum as a new user would.
2. Install into a path with spaces using an ordinary Windows user account.
3. Verify prerequisite installation and package downloads complete, then open the
   app from the Start menu and check the repair/help shortcuts.
4. Download the required models explicitly and generate/review a small real dataset.
5. Stop and reopen the project, resume it, and export accepted audio.
6. Check upgrade and uninstall while retaining a test project and its recordings.

No clean-machine runtime installation is implied by the automated tests. Keep that
distinction in release notes. The current build is unsigned; signing requires a
publisher certificate and is a separate distribution step.

Publish the setup executable and `.sha256` as release **Assets**, not as ordinary
Git-tracked binaries. `dist/` is intentionally ignored. Check the payload inventory
before upload. Building does not create a GitHub release, commit changes, or push.
