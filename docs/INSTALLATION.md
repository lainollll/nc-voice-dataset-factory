# Install NC Voice Dataset Factory on Windows

## Before you start

Use Windows 11 on a 64-bit Intel/AMD PC and a local NTFS drive. Speech generation
requires an NVIDIA CUDA-compatible GPU and a working NVIDIA driver; the installer
does not install or update graphics drivers. Viewing projects and exporting existing
recordings do not require speech generation.

Plan for at least 20 GB of free space initially for the app packages and models.
Recordings, original audio, exports, and optional LFM3 training need additional
space. This is a planning allowance, not a fixed download size or a GPU memory
requirement. Setup does not bundle model weights or your existing datasets.

Install or update **App Installer** from Microsoft Store if WinGet is unavailable.
Setup uses WinGet to install missing Python 3.11, Git, and FFmpeg. Prerequisite
installers may request Windows administrator approval.

## Using the setup executable

1. Download `NC-Voice-Dataset-Factory-<version>-Setup.exe` from the **Assets** of a
   [GitHub release](https://github.com/lainollll/nc-voice-dataset-factory/releases).
   The repository must be accessible to your GitHub account while it is private.
   If a release has no setup executable, use the source instructions below.
2. Double-click the executable. Read the installation information and select a
   writable application folder. The default is:

   ```text
   %LOCALAPPDATA%\Programs\NC Voice Dataset Factory
   ```

3. Optionally select a desktop shortcut, then click **Install**.
4. Keep the dependency window open. It creates an app-local `.venv` and downloads
   the pinned GUI, CUDA/TTS, transcription, export, and speaker packages. This can
   take a while on the first install; a small setup executable does not mean a
   small runtime download.
5. Wait for a successful finish. Choose **Open NC Voice Dataset Factory**, or
   launch it from the Start menu afterward.

The Start menu folder contains the app, **Repair setup**, and **Installation help**.
Dependency setup logs are in `<installation folder>\setup-logs\setup-*.log`.
The setup wizard also writes a `Setup Log*.txt` file in your Windows temporary
folder (`%TEMP%`) for file-copy and shortcut errors.

The installer is not code-signed. Windows may display an unknown-publisher or
reputation prompt. Check that the file came from the intended repository/release;
do not disable Windows security features to install it. A SHA-256 file accompanies
the setup executable to check download integrity.

## Installing from the repository ZIP

1. Click **Code → Download ZIP** on the repository page.
2. Use **Extract All**. Keep the application in a writable folder on a local NTFS
   drive, for example `C:\NCVoiceDatasetFactory`. Avoid `Program Files`, a ZIP
   viewer, or a folder managed by a cloud synchronization service.
3. Open the extracted folder that contains `setup.cmd` and `pyproject.toml`.
4. Double-click **setup.cmd**. It finds or installs the prerequisites and installs
   the app into that folder's `.venv`. Wait for **Setup finished**.
5. Double-click **Start_NC_Voice_Dataset_Factory.cmd** whenever you want to run it.

This source installation does not create shortcuts or a Windows uninstall entry.
Do not move an installed `.venv` between computers or folders. For a different
location, extract a fresh source folder and run setup there.

## First launch and model downloads

1. On **Getting Started**, choose your reference WAV, language, and a small target.
   Confirm that you have permission to use the voice.
2. Click **Inspect missing models** if requested. Review the model sources,
   licenses, and download sizes, then click **Download models**.
3. Once the required speech and faster-whisper transcription models are available,
   click **Generate dataset**.
4. Listen to accepted recordings in **Dataset Browser** before a larger batch.
   Export into a new folder when you are satisfied with the results.

To hear the reference voice first, use **Preview voice and Chatterbox settings…**
on Getting Started. The selected WAV and permission choice are carried into Voice
Manager. Find a local model, adjust the three main sliders, enter sample text, and
click **Generate voice preview**. **Replay**, **Stop**, and **Reference** control
playback. No project is required; standalone previews go into the app's `previews`
folder. Previewing does not save the voice/profile or add accepted recordings.

Save a project/profile to use the selected settings for new **Batch Generator**
jobs. Getting Started uses its automatic dataset settings. Existing saved jobs
retain their original settings when resumed.

For LFM3, follow the three numbered sections on **LFM3 Fine-tuning**. Choose your
recordings, prepare the separate training tools, then start training. Start remains
unavailable while data or tools are missing; the screen explains the next step.
Optional overrides are under **Advanced training options**. Logs and emergency
**Force stop…** are under **Technical details and logs**. A force stop asks for
confirmation because it can lose unsaved work. Prefer **Stop and save**.

Python packages and model weights are separate downloads. A successful installer
does not mean all model weights have been downloaded. Existing compatible local
models can be discovered by the app. A local language-model server is optional:
the built-in text source works without one.

## If installation fails

| What you see | What to do |
| --- | --- |
| WinGet or App Installer is missing | Install/update **App Installer** in Microsoft Store. Open a new setup window and retry. |
| A prerequisite installed but cannot be found | Close the setup window and rerun setup so it picks up the updated PATH. |
| Download, network, disk-space, or pip error | Fix the reported condition, then retry **Repair setup**. The existing app-local environment is reused. |
| Setup says the app is not ready | Open the most recent file in `setup-logs`. Resolve the first dependency error, then retry **Repair setup**. |
| The source launcher says setup is required | Run `setup.cmd` in the same extracted application folder, then launch again. |
| Wrong Python version / incomplete environment | Use a fresh installation folder. Setup requires 64-bit Python 3.11 and will not replace an incompatible existing environment automatically. |
| Models are missing | Use **Inspect missing models → Download models** inside the app. |
| CUDA is unavailable or GPU memory is insufficient | Check your NVIDIA driver and other GPU workloads. The app does not silently fall back to CPU when GPU generation is required. |

For an installer installation, **Repair setup** runs `Repair_Setup.cmd` and saves
a new transcript. You can also double-click that file in the installation folder.
For a source installation, rerun `setup.cmd`; its console keeps error details
visible. Do not manually mix the app's packages with NeuralCompanion's environment.

## Resume an interrupted dataset

1. Click **Create / Open project…** and select the original project folder.
   It contains `factory.sqlite3`, `audio`, and `project_config.json`.
2. Open **Batch Generator** and select the saved job.
3. Click **Resume**. Existing accepted audio is retained; interrupted work is
   recovered and processing continues toward the saved target.

Do not click **Generate dataset** to recover an old project: that creates a new
project. The Getting Started page's resume control is intended for its current
session; Batch Generator handles reopening and resuming saved jobs. Keep the
saved reference WAV and model directories in their original locations.

## Updates, backups, and uninstall

Close the app and stop any active generation or training before installing an
update. Back up project folders and the reference WAVs first. Run the newer
installer into the same application folder; it updates packaged app files and
reruns the pinned dependency setup. It does not replace your project databases.

For a source update, use a fresh extracted folder and run setup. Open your existing
project from there. Keep the old folder until you have confirmed any app-local
reference/model paths still point to available files.

An installer installation can be removed through **Windows Settings → Apps →
Installed apps → NC Voice Dataset Factory → Uninstall**. Uninstall removes files
that the installer copied and its shortcuts. It preserves files created afterward,
including `.venv`, `models`, `projects`, `lfm3-training`, exports, and setup logs.
The remaining folder can be large. Back up the data you want before deleting it
manually. Python, Git, and FFmpeg installed as prerequisites are not uninstalled.

For source installations, there is no uninstall program. Back up projects, model
weights, and reference audio before removing the extracted application folder.

See [the advanced user guide](USER_GUIDE.md) for export formats, worker settings,
optional LFM3 fine-tuning, diagnostics, and command-line use.
