# NC Voice Dataset Factory

Create voice training datasets from a reference WAV on Windows. The app generates
speech locally, checks each recording, and exports accepted audio with transcripts
for Hugging Face or LFM3 training. You can pause, stop, and resume saved projects.

## What you need

- **Windows 11, 64-bit Intel/AMD**, and a local **NTFS** drive.
- An **NVIDIA CUDA-compatible GPU** with a current NVIDIA driver for speech generation.
- Internet access during installation and the first model downloads.
- Free disk space for Python packages, model weights, and recordings. Plan for
  **at least 20 GB initially**, with more space for large datasets and fine-tuning.
- A reference **WAV** and permission to use that voice.

Python 3.11, Git, and FFmpeg are checked automatically by setup. Missing tools are
installed through WinGet. You do not need to install Python packages by hand.

## Install on Windows

### Option 1: Windows installer

1. [Download the Windows installer](https://github.com/lainollll/nc-voice-dataset-factory/releases/download/v0.1.0/NC-Voice-Dataset-Factory-0.1.0-Setup.exe).
   See [Releases](https://github.com/lainollll/nc-voice-dataset-factory/releases)
   for release notes and the SHA-256 checksum.
2. Double-click the installer and keep the default installation folder, or choose
   another writable folder on a local NTFS drive.
3. Click **Install**. Keep the dependency setup window open while packages
   download. Windows may ask for approval to install missing prerequisites.
4. When setup succeeds, open **NC Voice Dataset Factory** from the Start menu.

This is a small **online installer**. It downloads the required runtime during
setup. Speech/transcription model weights are downloaded separately inside the app.
If setup fails, use **Repair setup** in the Start menu; details are saved in the
installation folder's `setup-logs` directory.

### Option 2: Install from the source ZIP

1. On this repository's page, click **Code → Download ZIP**.
2. Right-click the ZIP and choose **Extract All**. Put the extracted folder somewhere
   writable, such as `C:\NCVoiceDatasetFactory`. Open the extracted folder containing
   `setup.cmd`; do not run it from inside the ZIP.
3. Double-click **setup.cmd** and wait until it says **Setup finished**. Keep the
   window open and approve prerequisite installations if Windows asks.
4. Double-click **Start_NC_Voice_Dataset_Factory.cmd** to open the app.

The source installation runs from that folder and does not add Start menu shortcuts.
It uses its own `.venv`; NeuralCompanion's existing environment is kept separate.

**Need help?** See [installation and troubleshooting](docs/INSTALLATION.md).
Maintainers can [build the Windows installer](docs/BUILD_INSTALLER.md) from this repo.

## Make your first dataset

1. On **Getting Started**, choose a reference WAV, select English or Swedish, and
   start with a small target such as **10 accepted recordings**.
2. Confirm that you have permission to use the voice.
3. If models are missing, click **Inspect missing models**, review the downloads,
   then click **Download models**. Wait for those downloads to finish.
4. Click **Generate dataset**. The app creates a project and checks generated
   recordings against the transcripts. Rejected clips do not count toward your target.
5. Listen to accepted clips in **Dataset Browser**, then use **Export** to write
   your dataset into a new destination folder.

The built-in text source needs no language model server. Optional LFM3 fine-tuning
has its own **Set up training environment** step after you have a dataset.

## Resume after stopping or a crash

Your accepted recordings and job settings are saved in the project folder.

1. Open the app and click **Create / Open project…**.
2. Select the original project folder containing `factory.sqlite3`.
3. Open **Batch Generator**, select the saved job, and click **Resume**.

Resume keeps accepted recordings and recovers interrupted samples using the job's
saved settings. After reopening the app, use **Batch Generator → Resume** for an
existing project. Keep the reference WAV and model folders available.

## Preview a voice with Chatterbox

Click **Preview voice and Chatterbox settings…** on Getting Started. Your selected
WAV is carried into Voice Manager. Use **Find local model**, confirm voice
permission, then adjust the delivery, expressiveness, and voice-guidance sliders.
Type a short sentence and click **Generate voice preview**; compare it using
**Replay**, **Stop**, and **Reference**.

You can preview without creating a project. Previews are saved separately and do
not count as dataset recordings. Save the voice/project to keep settings for new
**Batch Generator** jobs; Getting Started retains its automatic dataset settings.

## Train with LFM3

The **LFM3 Fine-tuning** page guides you through **Choose recordings → Prepare
training → Train and save**. It shows the selected voice, accepted count, and audio
duration, and explains what must be ready before Start becomes available.

The initial 200 training steps are a short test. Optional settings and technical
logs are expandable. Progress distinguishes loading, encoding, training, and saving.
After success, click **Open saved voice files**. Chatterbox previews test your
reference voice; they do not preview or activate the trained LFM3 adapter.

## More help

- [Installation, repair, updates, and uninstall](docs/INSTALLATION.md)
- [Advanced usage, export formats, training, and CLI commands](docs/USER_GUIDE.md)
- [Installer build and validation instructions](docs/BUILD_INSTALLER.md)
- [Voice authorization](docs/VOICE_AUTHORIZATION.md)
- [Recorded validation evidence](docs/VALIDATION.md)
