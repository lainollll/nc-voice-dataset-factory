"""Beginner walkthrough; actions delegate to the existing application pages."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class GuideStep:
    title: str
    page: str
    instructions: str
    button: str
    action: str = "page"


STEPS = (
    GuideStep(
        "Create a project",
        "Dashboard",
        """
        <p>Choose an empty folder for your first dataset. The app keeps your settings,
        generated recordings and review history together in that folder.</p>
        <ol><li>Click <b>Create / Open project</b> below and choose a local folder.</li>
        <li>Use a separate project for each voice or dataset you want to keep apart.</li></ol>
        <p>You can reopen the same folder later to continue your work.</p>
        """,
        "Create / Open project…",
        "project",
    ),
    GuideStep(
        "Prepare the models",
        "Voice Manager",
        """
        <p>You need a speech model to generate audio and an ASR model to check the words.</p>
        <ol><li>In <b>Voice Manager</b>, choose <b>Chatterbox English</b> for English,
        or <b>Chatterbox Multilingual V3</b> for Swedish or English.</li>
        <li>If you already have the matching weights, use <b>Choose model folder</b>.</li>
        <li>Otherwise, scroll down to the download controls. Choose <b>english</b> or
        <b>multilingual-v3</b>, then <b>Inspect model download</b> to review the source,
        license and size. <b>Download inspected model</b> saves it into an empty folder.</li>
        <li>Repeat with download kind <b>asr</b>. Its folder is filled in on
        <b>Quality Control</b> when the download finishes.</li></ol>
        <p>Downloads can be several GB. Inspecting a model shows repository information;
        downloading is a separate action. Keep the chosen speech model and its folder matched.</p>
        <p><b>Missing Python dependencies?</b> Close the app, double-click <b>setup.cmd</b>,
        wait for setup to finish, then reopen the app.</p>
        """,
        "Open Voice Manager",
    ),
    GuideStep(
        "Choose and test a voice",
        "Voice Manager",
        """
        <ol><li>Enter a voice name and use <b>Import reference WAV</b> to select your recording.</li>
        <li>Choose <b>English</b> or <b>Swedish</b>. Swedish needs a multilingual speech model.</li>
        <li>Confirm that you own or have permission to use the voice.</li>
        <li>Click <b>Save voice / project</b>, then <b>Generate voice preview</b>.</li>
        <li>Listen to the preview. Use <b>Play reference</b> to compare it with your recording.</li></ol>
        <p>A reference WAV is the voice sample, not the model folder. Both are needed.
        If generation reports a missing model, return to step 2 and check the selected folder.</p>
        """,
        "Open Voice Manager",
    ),
    GuideStep(
        "Prepare your sentences",
        "Text Generator",
        """
        <ol><li>Choose <b>Built-in corpus</b> for the easiest first run.</li>
        <li>Set <b>Preview sentence count</b> to <b>10</b>.</li>
        <li>Keep the initial word limits and category weights, then click <b>Generate preview</b>.</li>
        <li>Read or edit the sentences on <b>Batch Generator</b>.</li></ol>
        <p>The text language follows your selected voice. Later you can import your own
        TXT/CSV/JSONL file or choose <b>OpenAI-compatible LLM</b> for more variety.
        Enter the server's base URL, then choose a text model from the automatically
        populated dropdown. Use <b>Refresh models</b> to retry or update it.</p>
        <p>Built-in corpus uses bundled templates and word choices, so it works offline
        and has finite variety. Its category weights control the relative sentence mix.</p>
        <p>For large text sets, the editor shows the first 500 sentences and retains the rest for the job.</p>
        """,
        "Open Text Generator",
    ),
    GuideStep(
        "Check quality settings",
        "Quality Control",
        """
        <ol><li>Keep <b>Verify transcripts with faster-whisper</b> checked.</li>
        <li>Select your local faster-whisper model folder if it was not filled in during download.</li>
        <li>Keep <b>CPU / int8</b> for ASR to leave GPU memory available for speech generation.</li>
        <li>Start with maximum <b>WER 0.10</b>, <b>CER 0.10</b>, and clips lasting <b>3–15 seconds</b>.</li></ol>
        <p>WER and CER measure word and character differences; lower values are better.
        Failed checks keep a clip out of automatic acceptance.</p>
        <p>Independent speaker verification is optional. Leave it off for the first test;
        it requires a separate local speaker model and calibrated reference recordings.</p>
        """,
        "Open Quality Control",
    ),
    GuideStep(
        "Run a 10-clip test",
        "Batch Generator",
        """
        <ol><li>Use the button below to set the accepted-recording target and text-preview
        count to <b>10</b>, with the duration target turned off.</li>
        <li>Check the prepared sentences. Save your project if you changed voice settings.</li>
        <li>Scroll down and click <b>Start</b> on Batch Generator.</li>
        <li>Watch the accepted count and error messages. A rejection does not count toward the target.</li></ol>
        <p>Leave <b>Automatically choose GPU workers</b> and quality-check overlap enabled.
        Auto picks a worker count that fits available GPU memory, up to your chosen limit
        (maximum 8). For an exact count, uncheck Auto. If even one worker cannot fit,
        free GPU memory; keep transcript checking on CPU/int8 to leave room for speech.</p>
        <p>Auto also runs CPU transcript checks concurrently when CPU and RAM allow
        and speaker verification is off. Progress shows active generation tasks,
        quality checks and queued checks; Sampling bars alone do not show utilization.</p>
        <p><b>Pause</b> and <b>Stop</b> finish in-flight clips safely. <b>Resume</b> continues
        the selected job with its saved settings. A stopped job can still be reviewed and exported.</p>
        <p>The button below only prepares the settings; click Start yourself when ready.
        It will not change targets while another operation is active.</p>
        """,
        "Set up a 10-clip test",
        "small-test",
    ),
    GuideStep(
        "Listen and review",
        "Dataset Browser",
        """
        <ol><li>Open <b>Dataset Browser</b> and select a recording.</li>
        <li>Click <b>Play</b>. Read the training text and ASR result while listening.</li>
        <li>Use <b>Random accepted review</b> to spot-check accepted clips.</li>
        <li>Reject or retry clips you do not want. Review actions ask for a reason.</li></ol>
        <p>If you edit a transcript, save it and <b>Revalidate</b> before export.
        To create new audio for the edited words, use <b>Regenerate edited text</b>,
        then resume that sample's job.</p>
        <p>Automatic transcript checks help, but listening is still useful for pronunciation,
        voice consistency and naturalness.</p>
        """,
        "Open Dataset Browser",
    ),
    GuideStep(
        "Export and scale up",
        "Export",
        """
        <ol><li>Keep the default <b>LJ Speech-style Hugging Face Parquet</b> selected.
        Add other formats if needed. <b>Standard CSV</b> is a simple text/audio index;
        <b>LFM3 notebook loader + JSONL</b> prepares data for the Liquid training workflow.</li>
        <li>Keep the initial validation fraction of <b>0.10</b> for a roughly 90/10 split.</li>
        <li>Click <b>Export accepted dataset</b> and choose a new destination folder.</li>
        <li>Read the result. File integrity and transcript/audio pairing are checked before publication.</li></ol>
        <p>Rejected clips stay out of training splits. The optional rejected-audio export
        puts them in a separate folder.</p>
        <p>When your short test sounds right, return to Text Generator and Batch Generator
        to prepare more sentences and choose a larger accepted target. You can use any
        useful dataset size; exactly 1,000 recordings is not required.</p>
        <p>The default <b>LJ Speech-style Hugging Face Parquet</b> matches MikhailT/lj-speech.
        The <b>ljspeech_hf</b> subfolder contains a single <b>full</b> split with embedded
        22,050 Hz audio and file/spoken_text/normalized_text/audio columns.</p>
        <p>Use <b>LFM3 Fine-tuning</b> to run the official notebook from the app in its
        separate training environment.</p>
        """,
        "Open Export",
    ),
)

SIMPLE_STEPS = (
    GuideStep(
        "Choose WAV and dataset size",
        "Getting Started",
        """
        <ol><li>Click <b>Choose WAV</b> and select the voice recording you want to use.</li>
        <li>Choose <b>English</b> or <b>Swedish</b> and how many accepted recordings
        or minutes of audio you want.</li>
        <li>Confirm voice permission, then click <b>Generate dataset</b>.</li></ol>
        <p>The app finds matching installed models, prepares sentences, creates a new
        project, and sets GPU workers and quality checks automatically.</p>
        <p>If models are missing, click <b>Inspect missing models</b>. Review the size
        and source, click <b>Download models</b>, then Generate dataset.</p>
        <p>You do not need to fill in the other cards. The advanced walkthrough is
        available above if you want to customize the process.</p>
        """,
        "Open simple setup",
    ),
    GuideStep(
        "Optional: preview your voice",
        "Voice Manager",
        """
        <ol><li>Use <b>Preview voice and Chatterbox settings</b> on Getting Started
        to bring your chosen WAV into Voice Manager.</li>
        <li>Confirm voice permission and click <b>Find local model</b> if needed.
        Model folders and saved profiles are under the expandable details.</li>
        <li>Adjust <b>Variation in delivery</b> (temperature), <b>Expressiveness</b>
        (exaggeration), and <b>Voice guidance</b> (CFG). You can drag the sliders
        or type exact values.</li>
        <li>Enter a short sentence and click <b>Generate voice preview</b>. Use
        <b>Replay</b>, <b>Stop</b>, and <b>Reference</b> to compare the audio.</li></ol>
        <p>These are Chatterbox/reference-voice previews, not previews of a trained
        LFM3 adapter. Previewing does not create dataset records or save settings.</p>
        <p>Open or create a project and use <b>Save voice / project</b> to keep
        settings for new <b>Batch Generator</b> jobs. Getting Started uses its
        automatic dataset settings. Existing jobs keep their saved settings.</p>
        """,
        "Open Voice Manager",
    ),
    GuideStep(
        "Let the app generate",
        "Getting Started",
        """
        <p>Follow the accepted count and duration on Getting Started. Sentences are
        created as needed, and transcript and audio checks run automatically.</p>
        <p>Auto chooses GPU workers from free memory, up to 8, and shows the chosen count
        in progress. It also chooses CPU transcript-check concurrency from available CPU
        and RAM. Active tasks and queued checks appear below the accepted count.
        Starting again after a stopped job can choose different counts.</p>
        <p><b>Pause</b>, <b>Resume</b> and <b>Stop</b> use the existing safe job controls.
        In-flight recordings finish before stopping. Rejected recordings do not
        count toward your target.</p>
        <p>The project folder is shown below the progress. Reopen that folder from
        Dashboard and use Batch Generator to resume it after restarting the app.</p>
        """,
        "Show progress",
    ),
    GuideStep(
        "Listen and export",
        "Dataset Browser",
        """
        <p>Listen to a few accepted recordings in <b>Dataset Browser</b>.</p>
        <p>Then use <b>Export accepted dataset</b> on Getting Started and choose where
        to save it. The default export prepares LJ Speech-style Hugging Face Parquet
        with embedded 22,050 Hz audio for training.</p>
        <p>You can use the Export card to choose other formats if needed.</p>
        """,
        "Open Dataset Browser",
    ),
    GuideStep(
        "Optional: fine-tune LFM3",
        "LFM3 Fine-tuning",
        """
        <ol><li>In <b>1. Choose your recordings</b>, select the current project or a saved
        dataset export. Check the voice, accepted count and duration shown below it.</li>
        <li>In <b>2. Prepare training</b>, set up the separate training tools if they are
        missing. If it says Ready, you can continue. Repair is optional.</li>
        <li>In <b>3. Train and save your voice</b>, click <b>Start fine-tuning</b>.
        The first run downloads the LFM3 model and codec.</li></ol>
        <p>Start is available when recordings and tools are ready and other work has finished.
        The initial 200 steps are a short test; change the length under <b>Advanced training options</b>.</p>
        <p>Follow the loading, encoding, training and saving stages. <b>Stop and save</b>
        requests a safe stop. Logs and emergency <b>Force stop</b> are under
        <b>Technical details and logs</b>.</p>
        <p>After success, use <b>Open saved voice files</b>. The result is an LFM3
        adapter; Chatterbox previews use the reference voice separately.</p>
        """,
        "Open LFM3 Fine-tuning",
    ),
)


class GettingStartedGuide(QWidget):
    navigate_requested = Signal(str)
    project_requested = Signal()
    small_test_requested = Signal()

    def show_training_help(self) -> None:
        if self.mode.currentIndex() == 0:
            self.steps.setCurrentRow(len(SIMPLE_STEPS) - 1)

    def show_voice_preview_help(self) -> None:
        if self.mode.currentIndex() == 0:
            self.steps.setCurrentRow(next(i for i, step in enumerate(SIMPLE_STEPS) if step.page == "Voice Manager"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        self.mode = QComboBox()
        self.mode.addItems(["Simple setup", "Advanced walkthrough"])
        layout.addWidget(self.mode)
        self.counter = QLabel()
        self.counter.setObjectName("subtitle")
        layout.addWidget(self.counter)
        body = QVBoxLayout()
        self.steps = QListWidget()
        self.steps.setWordWrap(True)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.steps.setFixedHeight(150)
        body.addWidget(self.steps)
        details = QVBoxLayout()
        self.step_title = QLabel()
        self.step_title.setWordWrap(True)
        self.step_title.setStyleSheet("font-size:20px; font-weight:600;")
        details.addWidget(self.step_title)
        self.instructions = QTextBrowser()
        self.instructions.setOpenExternalLinks(False)
        self.instructions.setMinimumHeight(180)
        details.addWidget(self.instructions, 1)
        self.action_button = QPushButton()
        self.action_button.setObjectName("primary")
        self.action_button.clicked.connect(self._activate)
        details.addWidget(self.action_button)
        body.addLayout(details, 1)
        layout.addLayout(body, 1)
        footer = QHBoxLayout()
        self.back_button, self.next_button = QPushButton("Back"), QPushButton("Next step")
        self.back_button.clicked.connect(lambda: self.steps.setCurrentRow(self.steps.currentRow() - 1))
        self.next_button.clicked.connect(lambda: self.steps.setCurrentRow(self.steps.currentRow() + 1))
        footer.addWidget(self.back_button)
        footer.addStretch()
        footer.addWidget(self.next_button)
        layout.addLayout(footer)
        note = QLabel("Keep this guide beside any card, or use Detach to read it in a separate window.")
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        layout.addWidget(note)
        self.steps.currentRowChanged.connect(self._show_step)
        self.mode.currentIndexChanged.connect(self._change_mode)
        self._change_mode(0)

    def _change_mode(self, index: int) -> None:
        self._guide_steps = SIMPLE_STEPS if index == 0 else STEPS
        self.steps.blockSignals(True)
        self.steps.clear()
        for number, step in enumerate(self._guide_steps, 1):
            self.steps.addItem(f"{number}. {step.title}")
        self.steps.blockSignals(False)
        self.steps.setCurrentRow(0)

    def _show_step(self, index: int) -> None:
        if not 0 <= index < len(self._guide_steps):
            return
        step = self._guide_steps[index]
        self.counter.setText(f"Step {index + 1} of {len(self._guide_steps)}")
        self.step_title.setText(step.title)
        self.instructions.setHtml(step.instructions)
        self.action_button.setText(step.button)
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self._guide_steps) - 1)

    def _activate(self) -> None:
        step = self._guide_steps[self.steps.currentRow()]
        if step.action == "project":
            self.project_requested.emit()
        elif step.action == "small-test":
            self.small_test_requested.emit()
        else:
            self.navigate_requested.emit(step.page)


class GettingStartedDock(QDockWidget):
    """Keep the single guide available while navigating, docked or floating."""

    def __init__(self, guide: GettingStartedGuide, parent: QWidget) -> None:
        super().__init__("Getting Started", parent)
        self.setObjectName("gettingStartedDock")
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.setMinimumWidth(340)
        self.setWidget(guide)
        header = QWidget(self)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 6, 8, 6)
        title = QLabel("Getting Started")
        title.setStyleSheet("font-size:16px; font-weight:600;")
        layout.addWidget(title)
        layout.addStretch()
        self.detach_button = QPushButton("Detach")
        self.detach_button.clicked.connect(self._toggle_floating)
        layout.addWidget(self.detach_button)
        self.setTitleBarWidget(header)
        self.topLevelChanged.connect(self._floating_changed)
        self._floating_changed(False)

    @Slot()
    def _toggle_floating(self) -> None:
        self.setFloating(not self.isFloating())
        self.show()

    @Slot(bool)
    def _floating_changed(self, floating: bool) -> None:
        self.detach_button.setText("Dock" if floating else "Detach")
        self.detach_button.setToolTip(
            "Return instructions to the right of the cards"
            if floating
            else "Read instructions in a separate window while using the cards"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        # The guide stays available; the floating window's close action returns it.
        event.ignore()
        self.setFloating(False)
        self.show()
