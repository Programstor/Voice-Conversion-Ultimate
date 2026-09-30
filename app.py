import sys
import os
from bin.config import SR_PROFILES, ST_SRS, load_config
from bin.log import get_logger, setup_logging
from bin.modules import Modules

cfg = load_config()
cfg.paths.ensure()
setup_logging(cfg.paths.logs)
log = get_logger("app")

import sounddevice as sd
import re
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QComboBox, QPushButton,
                             QLineEdit, QFileDialog, QLabel, QCheckBox, QGroupBox,
                             QRadioButton, QButtonGroup, QSizePolicy)

from config.stylesheet import get_stylesheet, update_state
from lib.ui.lslider import LabeledSlider, LiveParams
from lib.ui.player import AudioPlayer
from lib.ui.sbutton import TaskManager
from lib.ui.lchart import LossChart, ChartBridge


class VCU_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = cfg
        self.wdgPlayer = None
        self.task_manager = TaskManager(cfg)
        self.modules = Modules(cfg)
        self.setWindowTitle("Voice Conversion - Ultimate")
        self.resize(1200, 600)
        self.showMaximized()
        self.raise_()
        self.activateWindow()

        # Layout
        wdgCenter = QWidget()
        self.setCentralWidget(wdgCenter)
        self.layMain = QVBoxLayout(wdgCenter)
        self.wdgChart = LossChart()
        self.chartBridge = ChartBridge(self.wdgChart)

        self.twdgTabs = QTabWidget()
        self.twdgTabs.addTab(self.init_inference_tab(), "Inference")
        self.twdgTabs.addTab(self.init_training_tab(), "Training")
        self.twdgTabs.addTab(self.init_realtime_tab(), "Realtime")
        self.layMain.addWidget(self.twdgTabs)

        self.lblStatus = QLabel("System Ready")
        self.lblStatus.setWordWrap(False)
        self.lblStatus.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lblStatus.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.layMain.addWidget(self.lblStatus)

        # Postprocess
        self.models = []
        self.indices = []
        self.refresh_voice_models()
        self.refresh_audio_devices()

    @staticmethod
    def _repopulate(combo: QComboBox, items, keep=None):
        """Replace a combo's items without emitting currentIndexChanged, restoring `keep` if present."""
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(items)
            if keep in items:
                combo.setCurrentIndex(items.index(keep))
        finally:
            combo.blockSignals(False)

    @staticmethod
    def _set_index(combo: QComboBox, index: int):
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)

    def refresh_audio_devices(self):
        try:
            devices = sd.query_devices()

            self.qboxInput.clear()
            self.qboxOutput.clear()

            junk_keywords = ["mapping", "mapper", "primary", "system32"]

            for i, dev in enumerate(devices):
                name_low = dev['name'].lower()
                if any(junk in name_low for junk in junk_keywords): continue
                if dev['hostapi'] != sd.default.hostapi: continue

                # Format: "0 | Microphone (Realtek)"
                display_name = f"{i} | {dev['name']}"

                if dev['max_input_channels'] > 0:
                    self.qboxInput.addItem(display_name)

                if dev['max_output_channels'] > 0:
                    self.qboxOutput.addItem(display_name)

            self.qboxInput.setMinimumWidth(280)
            self.qboxOutput.setMinimumWidth(280)

        except Exception as e:
            log.exception("Error listing audio devices")
            # self.lblStatus.setText(f"Error: {e}")
            self._set_status(f"Error: {e}")

    def refresh_voice_models(self):
        try:
            keep = self.qboxVoice.currentText()
            self.models = sorted(f.name for f in self.cfg.paths.models.glob("*.pth")) or ["No models found!"]
            self.indices = sorted(f.name for f in self.cfg.paths.indices.glob("*.index")) or ["No indices found!"]

            self._repopulate(self.qboxVoice, self.models, keep)
            self._repopulate(self.qboxModel, self.models, keep)
            self._repopulate(self.qboxIndex, self.indices)
            self._repopulate(self.qboxFeatures, self.indices)
            self.refresh_index("audio")
        except Exception as e:
            log.exception("Error scanning models")
            self.lblStatus.setText(f"Error scanning models: {e}")

    def refresh_index(self, context):
        src, dst = ((self.qboxVoice, self.qboxModel) if context == "audio"
                    else (self.qboxModel, self.qboxVoice))
        selected_model = src.currentText()
        if not selected_model or "No models found" in selected_model:
            return

        self._set_index(dst, src.currentIndex())

        model_name_no_ext = os.path.splitext(selected_model)[0]
        # found_indices = [i for i in self.indices if model_name_no_ext in i]
        found_indices = [i for i in self.indices if re.search(rf'{re.escape(model_name_no_ext)}(?=[_.]|$)', i)]

        if found_indices:
            items = found_indices
            # self.lblStatus.setText(f"Found {len(found_indices)} index file(s) for {selected_model}")
            self._set_status(f"Found {len(found_indices)} index file(s) for {selected_model}")
        else:
            items = [f"No index file found for {selected_model}"]
            # self.lblStatus.setText(f"Warning: No index file found for {selected_model}")
            self._set_status(f"Warning: No index file found for {selected_model}")
        self._repopulate(self.qboxIndex, items)
        self._repopulate(self.qboxFeatures, items)

        self.task_manager.run_task(
            self.btnUnload, self.lblStatus,
            self.modules.load_model,
            ### Arguments ###
            gen_path=self.qboxVoice.currentText(),
            idx_path=self.qboxIndex.currentText()
        )

    def browse_file(self, field):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            str(self.cfg.paths.seperated)
        )
        if f:
            field.setText(f)

    def browse_folder(self, field):
        f = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            str(self.cfg.paths.root),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        if f:
            field.setText(f)

    def load_song(self, path):
        try:
            if self.wdgPlayer is not None:
                self.wdgPlayer.load_audio(path)
        except Exception as e:
            log.exception("Error loading audio into the player")
            # self.lblStatus.setText(f"Error loading audio: {e}")
            self._set_status(f"Error loading audio: {e}")

    def _set_status(self, text: str) -> None:
        self.lblStatus.setText(text)
        self.lblStatus.setToolTip(text)

    # ==========================================
    # TAB 1: INFERENCE
    # ==========================================
    def init_inference_tab(self):
        ic = self.cfg.inference
        pnlLeft = QGroupBox("Voice Options")
        lblVoice = QLabel("Voice model (.pth):")
        self.qboxVoice = QComboBox()
        self.qboxVoice.currentIndexChanged.connect(lambda: self.refresh_index("audio"))
        lblIndex = QLabel("Index model (.index):")
        self.qboxIndex = QComboBox()
        btnRefresh = QPushButton("Refresh Models")
        btnRefresh.clicked.connect(self.refresh_voice_models)
        lblAudio = QLabel("Audio file:")
        self.txtSong = QLineEdit()
        btnSong = QPushButton("Select Audio")
        btnSong.clicked.connect(lambda: self.browse_file(self.txtSong))
        self.btnUnload = QPushButton("Unload Voice")
        self.btnUnload.clicked.connect(
            lambda: self.task_manager.run_task(
                self.btnUnload,
                self.lblStatus,
                self.modules.unload_model
            )
        )
        pnlRight = QGroupBox("Inference Options")
        sldTranspose = LabeledSlider("Transpose (make voice more masculine / feminine):", -12, 12, ic.f0_up_key, " semitones")
        sldVolume = LabeledSlider("Volume Scaling (closer to 100 mimics the original):", 0, 100, ic.vol_scale*100, "%", 1, 0.01)
        sldProtect = LabeledSlider("Voiceless Protection (lower protects breaths/consonants more):", 0, 50, ic.protect*100, "%", 1, 0.01)
        sldRate = LabeledSlider("Index Rate (how closely the voice dialect is followed):", 0, 100, ic.idx_rate*100, "%", 1, 0.01)
        lblResample = QLabel("Resample to (set 0 to disable):")
        qboxResample = QComboBox()
        qboxResample.addItems(ST_SRS)
        cbxStitch = QCheckBox("Stitch Instruments")
        cbxStitch.setChecked(True)
        cbxStereo = QCheckBox("Keep Stereo")
        cbxStereo.setChecked(True)

        btnInfer = QPushButton("Infer Audio")
        update_state(btnInfer, "state", "active")
        btnInfer.setStyleSheet("font-size: 32px; font-weight: bold")
        btnInfer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        btnInfer.clicked.connect(
            lambda: self.task_manager.run_task(
                btnInfer,
                self.lblStatus,
                self.modules.infer_audio,
                callback=lambda path: self.load_song(path),  # type: ignore
                ### Arguments ###
                gen_path=self.qboxVoice.currentText(),
                idx_path=self.qboxIndex.currentText(),
                audio_path=self.txtSong.text(),
                f0_up_key=sldTranspose.value(),
                vol_scale=sldVolume.value(),
                protect=sldProtect.value(),
                index_rate=sldRate.value(),
                res_sr=int(qboxResample.currentText()),
                do_instruments=cbxStitch.isChecked(),
                keep_stereo=cbxStereo.isChecked()
            )
        )
        self.wdgPlayer = AudioPlayer()

        ## Master Layout
        page = QWidget()

        layMaster = QVBoxLayout(page)
        layTop = QHBoxLayout()
        layLeft = QVBoxLayout(pnlLeft)
        layRight = QVBoxLayout(pnlRight)
        layRBottom = QHBoxLayout()
        layBottom = QHBoxLayout()

        layLeft.addWidget(lblVoice)
        layLeft.addWidget(self.qboxVoice)
        layLeft.addWidget(lblIndex)
        layLeft.addWidget(self.qboxIndex)
        layLeft.addWidget(btnRefresh)
        layLeft.addWidget(lblAudio)
        layLeft.addWidget(self.txtSong)
        layLeft.addWidget(btnSong)
        layLeft.addWidget(self.btnUnload)

        layRight.addWidget(sldTranspose)
        layRight.addWidget(sldVolume)
        layRight.addWidget(sldProtect)
        layRight.addWidget(sldRate)
        layRight.addWidget(lblResample)
        layRight.addWidget(qboxResample)
        layRBottom.addWidget(cbxStitch)
        layRBottom.addWidget(cbxStereo)
        layRight.addLayout(layRBottom)

        layTop.addWidget(pnlLeft, 1)
        layTop.addWidget(pnlRight, 1)

        layBottom.addWidget(btnInfer, 1)
        layBottom.addWidget(self.wdgPlayer, 1)

        layMaster.addLayout(layTop, 80)
        layMaster.addLayout(layBottom, 20)

        return page

    # ==========================================
    # TAB 2: TRAINING
    # ==========================================
    def init_training_tab(self):
        pnlOptions = QGroupBox("Model Options")
        lblName = QLabel("Model name:")
        txtName = QLineEdit()
        lblFolder = QLabel("Training folder:")
        txtFolder = QLineEdit()
        txtFolder.setText(str(self.cfg.paths.dataset))
        btnChoose = QPushButton("Select Folder")
        btnChoose.clicked.connect(lambda: self.browse_folder(txtFolder))
        lblSample = QLabel("Sample rate:")
        self.grpSample = QButtonGroup(self)
        # One radio button per validated SRProfile (32 kHz has no validated architecture, so it is not offered)
        sample_buttons = {}
        for sr in sorted(SR_PROFILES):
            rb = QRadioButton(f"{sr // 1000} KHz")
            self.grpSample.addButton(rb)
            sample_buttons[sr] = rb
        sample_buttons[max(SR_PROFILES)].setChecked(True)
        selected_sr = lambda: next(sr for sr, rb in sample_buttons.items() if rb.isChecked())

        pnlSettings = QGroupBox("Training Options")
        sldEpochs = LabeledSlider("Total Epochs (number of passes through the dataset):", 0, 1000, cfg.train.epochs, "", 10)
        sldFrequency = LabeledSlider("Save Frequency (how frequently checkpoints are saved):", 5, 100, cfg.train.frequency, "", 5)
        sldBatch = LabeledSlider("Batch Size (how many chunks should be processed together):", 1, 40, cfg.train.batches)

        btnPreprocess = QPushButton("Preprocess")
        btnPreprocess.clicked.connect(
            lambda: self.task_manager.run_task(
                btnPreprocess,
                self.lblStatus,
                self.modules.preprocess_dataset,
                ### Arguments ###
                mo_name=txtName.text(),
                ds_path=txtFolder.text(),
                tgt_sr=selected_sr()
            )
        )
        btnTrain = QPushButton("Train Model")
        btnTrain.clicked.connect(
            lambda: self.task_manager.run_task(
                btnTrain,
                self.lblStatus,
                self.modules.train_model,
                ### Arguments ###
                mo_name=txtName.text(),
                epochs=sldEpochs.value(),
                frequency=sldFrequency.value(),
                batch=sldBatch.value(),
                chart=self.chartBridge      # thread-safe proxy for self.wdgChart
            )
        )
        btnIndex = QPushButton("Train Index")
        btnIndex.clicked.connect(
            lambda: self.task_manager.run_task(
                btnIndex,
                self.lblStatus,
                self.modules.train_index,
                ### Arguments ###
                mo_name=txtName.text()
            )
        )
        btnSequence = QPushButton("Sequence")
        update_state(btnSequence, "state", "active")
        btnSequence.setStyleSheet("font-weight: bold")

        ## Master Layout
        page = QWidget()

        layMaster = QVBoxLayout(page)
        layTop = QHBoxLayout()
        layOptions = QVBoxLayout(pnlOptions)
        laySettings = QVBoxLayout(pnlSettings)
        layActions = QHBoxLayout()
        laySample = QHBoxLayout()

        layOptions.addWidget(lblName)
        layOptions.addWidget(txtName)
        layOptions.addWidget(lblFolder)
        layOptions.addWidget(txtFolder)
        layOptions.addWidget(btnChoose)
        layOptions.addWidget(lblSample)
        for rb in sample_buttons.values():
            laySample.addWidget(rb)
        layOptions.addLayout(laySample)

        laySettings.addWidget(sldEpochs)
        laySettings.addWidget(sldFrequency)
        laySettings.addWidget(sldBatch)

        layActions.addWidget(btnPreprocess)
        layActions.addWidget(btnTrain)
        layActions.addWidget(btnIndex)
        layActions.addWidget(btnSequence)

        layTop.addWidget(pnlOptions, 1)
        layTop.addWidget(pnlSettings, 1)

        layMaster.addLayout(layTop, 40)
        layMaster.addLayout(layActions, 10)
        layMaster.addWidget(self.wdgChart, 50)

        return page

    # ==========================================
    # TAB 3: REALTIME
    # ==========================================
    def init_realtime_tab(self):
        ic = self.cfg.inference
        rt = self.cfg.realtime
        self.live_params = LiveParams(
            f0_up_key=ic.f0_up_key,
            vol_scale=ic.vol_scale,
            protect=ic.protect,
            index_rate=ic.idx_rate,
            gate_db=rt.gate_db
        )
        pnlAudio = QGroupBox("Voice Options")
        lblVoice = QLabel("Voice model (.pth):")
        self.qboxModel = QComboBox()
        self.qboxModel.currentIndexChanged.connect(lambda: self.refresh_index("input"))
        lblIndex = QLabel("Index model (.index):")
        self.qboxFeatures = QComboBox()
        btnRefreshM = QPushButton("Refresh Models")
        btnRefreshM.clicked.connect(self.refresh_voice_models)
        lblInput = QLabel("Microphone device:")
        self.qboxInput = QComboBox()
        lblOutput = QLabel("Speaker device:")
        self.qboxOutput = QComboBox()
        btnRefreshD = QPushButton("Refresh Devices")
        btnRefreshD.clicked.connect(self.refresh_audio_devices)

        pnlBackend = QGroupBox("Inference Options")
        sldTranspose = LabeledSlider("Transpose (make voice more masculine / feminine):", -12, 12, ic.f0_up_key, " semitones",
                             on_change=lambda v: self.live_params.set("f0_up_key", v))

        sldVolume = LabeledSlider("Volume Scaling (closer to 100 mimics the original):", 0, 100, ic.vol_scale*100, "%", 1, 0.01,
                                on_change=lambda v: self.live_params.set("vol_scale", v))

        sldProtect = LabeledSlider("Voiceless Protection (lower protects breaths/consonants more):", 0, 50, ic.protect*100, "%", 1, 0.01,
                                on_change=lambda v: self.live_params.set("protect", v))

        sldRate = LabeledSlider("Index Rate (how closely the voice dialect is followed):", 0, 100, ic.idx_rate*100, "%", 1, 0.01,
                                on_change=lambda v: self.live_params.set("index_rate", v))

        # Block length is a structural param; it stays static during the run
        sldBlock = LabeledSlider("Block Length (audio chunk inferred per step):", 50, 5000, rt.block_ms, " ms", 50)

        sldThreshold = LabeledSlider("Noise Gate (how quiet a sound should be registered):", -60, 0, int(rt.gate_db), " db",
                                    on_change=lambda v: self.live_params.set("gate_db", v))

        btnInfer = QPushButton("Infer Voice")
        update_state(btnInfer, "state", "active")
        btnInfer.setCheckable(True)
        btnInfer.clicked.connect(
            lambda: self.task_manager.run_task(
                btnInfer,
                self.lblStatus,
                self.modules.infer_input,
                live=True,
                ### Arguments ###
                gen_path=self.qboxModel.currentText(),
                idx_path=self.qboxFeatures.currentText(),
                live_params=self.live_params,
                block_ms=sldBlock.value(),
                mic_device=self.qboxInput.currentText(),
                spk_device=self.qboxOutput.currentText()
            )
        )

        ## Master Layout
        page = QWidget()

        layMaster = QVBoxLayout(page)
        layTop = QHBoxLayout()
        layAudio = QVBoxLayout(pnlAudio)
        layBackend = QVBoxLayout(pnlBackend)

        layAudio.addWidget(lblVoice)
        layAudio.addWidget(self.qboxModel)
        layAudio.addWidget(lblIndex)
        layAudio.addWidget(self.qboxFeatures)
        layAudio.addWidget(btnRefreshM)
        layAudio.addWidget(lblInput)
        layAudio.addWidget(self.qboxInput)
        layAudio.addWidget(lblOutput)
        layAudio.addWidget(self.qboxOutput)
        layAudio.addWidget(btnRefreshD)

        layBackend.addWidget(sldTranspose)
        layBackend.addWidget(sldVolume)
        layBackend.addWidget(sldProtect)
        layBackend.addWidget(sldRate)
        layBackend.addWidget(sldBlock)
        layBackend.addWidget(sldThreshold)

        layTop.addWidget(pnlAudio, 1)
        layTop.addWidget(pnlBackend, 1)

        layMaster.addLayout(layTop, 80)
        layMaster.addWidget(btnInfer, 20)

        return page


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet(str(cfg.paths.style)))

    window = VCU_GUI()
    window.show()
    sys.exit(app.exec())