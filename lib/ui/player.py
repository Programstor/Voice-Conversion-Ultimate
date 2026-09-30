import os
import numpy as np
from lib.audio.audio_io import load_audio_playback
import sounddevice as sd
from PyQt6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QMouseEvent
from config.stylesheet import update_state

class ClickSlider(QSlider):
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ratio = ev.position().x() / self.width()
            val = int(self.minimum() + ratio * (self.maximum() - self.minimum()))
            super().mousePressEvent(ev)
            self.setValue(val)
            self.sliderMoved.emit(val)
        else:
            super().mousePressEvent(ev)


class AudioPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_file = None
        self.is_sliding = False
        self._audio_data = None
        self._sr = 0
        self._current_frame = 0
        self._total_frames = 0
        self._duration_ms = 0
        self._is_playing = False
        self._stream = None
        self._volume = 0.5
        self._channels = 1

        self.init_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._sync_position)

    def init_ui(self):
        pnlPlayer = QGroupBox("Audio Player")
        self.lblSong = QLabel()
        self.lblSong.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sldProgress = ClickSlider(Qt.Orientation.Horizontal)
        self.sldProgress.setRange(0, 0)
        self.sldProgress.sliderPressed.connect(self.handle_slider_pressed)
        self.sldProgress.sliderReleased.connect(self.handle_slider_released)
        self.sldProgress.sliderMoved.connect(self.handle_seek)

        self.lblTime = QLabel("00:00 / 00:00")
        self.lblTime.setMinimumWidth(140)
        self.btnPlay = QPushButton()
        self.btnPlay.setFixedSize(60, 30)
        update_state(self.btnPlay, "state", "disabled")
        self.btnPlay.setEnabled(False)
        self.btnPlay.clicked.connect(self.toggle_playback)
        self.lblVolume = QLabel("50%") 
        self.lblVolume.setFixedWidth(60)
        self.lblVolume.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.sldVolume = ClickSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setRange(0, 100)
        self.sldVolume.setValue(50)
        self.set_volume(50)
        self.sldVolume.setFixedWidth(80)
        self.sldVolume.valueChanged.connect(self.set_volume)
        
        layMain = QVBoxLayout(self)
        layMain.setContentsMargins(0, 0, 0, 0)
        layPlayer = QVBoxLayout(pnlPlayer)
        layPlayer.setContentsMargins(20, 0, 20, 0)
        layControls = QHBoxLayout()

        layPlayer.addWidget(self.lblSong)
        layPlayer.addWidget(self.sldProgress)

        layControls.addWidget(self.lblTime)
        layControls.addStretch()
        layControls.addWidget(self.btnPlay)
        layControls.addStretch()
        layControls.addWidget(self.lblVolume)
        layControls.addWidget(self.sldVolume)
        layPlayer.addLayout(layControls)

        layMain.addWidget(pnlPlayer)

    def _sync_position(self):
        if self._sr > 0:
            pos_ms = int((self._current_frame / self._sr) * 1000)
            
            if self._current_frame >= self._total_frames and self._is_playing:
                self.toggle_playback()
                self._current_frame = 0
                pos_ms = 0
                
            self.update_position(pos_ms)

    def _audio_callback(self, outdata, frames, time, status):
        if not self._is_playing or self._audio_data is None:
            outdata.fill(0)
            return

        chunk_size = min(frames, self._total_frames - self._current_frame)
        if chunk_size == 0:
            outdata.fill(0)
            return

        outdata[:chunk_size] = self._audio_data[self._current_frame : self._current_frame + chunk_size] * self._volume
        
        if chunk_size < frames:
            outdata[chunk_size:].fill(0)

        self._current_frame += chunk_size

    def format_time(self, ms):
        seconds = (ms // 1000) % 60
        minutes = (ms // (1000 * 60)) % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _resample(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        num_samples = int(len(audio) * target_sr / orig_sr)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, num_samples)
        if audio.ndim == 1:
            return np.interp(x_new, x_old, audio).astype(audio.dtype)
        else:
            resampled = np.zeros((num_samples, audio.shape[1]), dtype=audio.dtype)
            for c in range(audio.shape[1]):
                resampled[:, c] = np.interp(x_new, x_old, audio[:, c])
            return resampled

    def load_audio(self, file_path):
        normalized_path = os.path.normpath(file_path)
        if not file_path or normalized_path == self.current_file:
            return

        self.cleanup()

        # 1. Separate file decoding phase
        try:
            y, sr = load_audio_playback(normalized_path)
        except Exception as e:
            self.lblSong.setText("Invalid file!")
            update_state(self.lblSong, "state", "error")
            update_state(self.btnPlay, "state", "disabled")
            self.btnPlay.setEnabled(False)
            self.sldProgress.setRange(0, 0)
            self.lblTime.setText("00:00 / 00:00")
            self._audio_data = None
            self._total_frames = 0
            self._current_frame = 0
            self._duration_ms = 0
            raise Exception(f"File decode error: {e}") from None

        # 2. Separate audio device setup phase
        try:
            if y.ndim == 1:
                audio_data = y.reshape(-1, 1)
            else:
                audio_data = y

            if audio_data.shape[1] > 2:
                audio_data = audio_data[:, :2]

            try:
                dev_info = sd.query_devices(kind='output')
                device_sr = int(dev_info['default_samplerate'])
                max_ch = int(dev_info['max_output_channels'])
            except Exception:
                device_sr = sr
                max_ch = 2

            if audio_data.shape[1] > max_ch:
                audio_data = audio_data[:, :max_ch]

            target_sr = sr
            if device_sr > 0 and abs(sr - device_sr) > 0:
                try:
                    sd.check_output_settings(samplerate=sr, channels=audio_data.shape[1])
                except Exception:
                    audio_data = self._resample(audio_data, sr, device_sr)
                    target_sr = device_sr

            self._audio_data = audio_data
            self._sr = target_sr
            self._total_frames = len(self._audio_data)
            self._current_frame = 0
            self._duration_ms = int((self._total_frames / self._sr) * 1000)

            # Do NOT open the stream here: opening it during a task-completion
            # callback briefly locks the audio device, which pauses any currently
            # playing stream and produces the audible stutter. Open it lazily in
            # toggle_playback() the first time the user hits Play.
            self._channels = self._audio_data.shape[1]
            self.current_file = normalized_path

            file_name = os.path.basename(normalized_path)
            self.lblSong.setText(f"{file_name:.50s}")

            update_state(self.lblSong, "state", "success")
            update_state(self.btnPlay, "state", "active")
            self.btnPlay.setEnabled(True)
            self.update_duration(self._duration_ms)

        except Exception as e:
            self.lblSong.setText("Audio device error!")
            update_state(self.lblSong, "state", "error")
            update_state(self.btnPlay, "state", "disabled")
            self.btnPlay.setEnabled(False)
            self.sldProgress.setRange(0, 0)
            self.lblTime.setText("00:00 / 00:00")
            self._audio_data = None
            self._total_frames = 0
            self._current_frame = 0
            self._duration_ms = 0
            raise Exception(f"Audio device error: {e}") from None

    def update_position(self, position):
        if not self.is_sliding:
            self.sldProgress.setValue(position)
        self.update_time_label(position, self._duration_ms)

    def update_duration(self, duration):
        self.sldProgress.setRange(0, duration)
        current = 0
        if self._sr > 0:
            current = int((self._current_frame / self._sr) * 1000)
        self.update_time_label(current, duration)

    def update_time_label(self, current, total):
        self.lblTime.setText(f"{self.format_time(current)} / {self.format_time(total)}")

    def _open_stream(self):
        if self._stream is None and self._audio_data is not None:
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=self._channels,
                callback=self._audio_callback,
            )

    def toggle_playback(self):
        if self._is_playing:
            self._is_playing = False
            if self._stream is not None:
                self._stream.stop()
            self._timer.stop()
            update_state(self.btnPlay, "state", "active")
        else:
            if self._current_frame >= self._total_frames:
                self._current_frame = 0
            self._open_stream()          # lazy: device claimed only when Play is pressed
            self._is_playing = True
            if self._stream is not None:
                self._stream.start()
            self._timer.start()
            update_state(self.btnPlay, "state", "pressed")

    def handle_slider_pressed(self):
        self.is_sliding = True

    def handle_slider_released(self):
        self.is_sliding = False
        self.handle_seek(self.sldProgress.value())

    def handle_seek(self, position):
        if self._sr > 0:
            target_frame = int((position / 1000.0) * self._sr)
            self._current_frame = min(target_frame, self._total_frames)

    def set_volume(self, volume):
        self._volume = volume / 100.0
        self.update_volume_display(volume)

    def update_volume_display(self, value):
        self.lblVolume.setText(f"{value}%")
        if value == 0:
            update_state(self.lblVolume, "state", "error")
        else:
            update_state(self.lblVolume, "state", "success")

    def cleanup(self):
        self._timer.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._is_playing = False
        self._channels = 1
        self.sldProgress.setValue(0)
        self.lblTime.setText("00:00 / 00:00")

    def closeEvent(self, a0):
        self.cleanup()
        super().closeEvent(a0)