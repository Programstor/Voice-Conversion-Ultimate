"""
lib/ui/sbutton.py - the button-driven task runner.

  UniversalWorker : runs one Modules method on a QThread. If the method accepts a
                    `stop_check` argument (or **kwargs), it is handed a callable the button
                    can flip to request a clean stop; TaskCancelled from the action is treated
                    as a normal stop, not an error.
  TaskManager     : wires a button + a shared status label to one task at a time.

Button text while its own task is running:
    live=True  (realtime): "Stop <original text>"    - overlay only pulses, never shows progress
    live=False (everything else): "Cancel <original text>" - overlay fills as a progress bar

Clicking a DIFFERENT button while a task is running does not queue anything: that button
shows "Busy with <the active button's text>" for `cfg.ui.busy_message_ms`, then reverts.
"""
import inspect

from PyQt6.QtCore import (QEasingCurve, QEvent, QObject, QSequentialAnimationGroup, Qt,
                          QThread, QTimer, QVariantAnimation, pyqtSignal)
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath
from PyQt6.QtWidgets import QLabel, QPushButton, QWidget

from bin.config import TaskCancelled, UiConfig
from bin.log import get_logger

log = get_logger("ui.task")


class UniversalWorker(QThread):
    progress_updated = pyqtSignal(int, int, str)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    cancelled = pyqtSignal(str)

    def __init__(self, action, *args, **kwargs):
        super().__init__()
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.keep_running = True

    def stop(self) -> None:
        """Called by TaskManager on a Stop/Cancel click. Only takes effect if `action` actually
        checks stop_check(); some quick actions (e.g. unload) have nothing to check and just run
        to completion."""
        self.keep_running = False

    def run(self) -> None:
        spec = inspect.getfullargspec(self.action)
        if "stop_check" in spec.args or spec.varkw is not None:
            self.kwargs["stop_check"] = lambda: not self.keep_running
        self.kwargs["progress_callback"] = self.progress_updated.emit

        name = getattr(self.action, "__name__", str(self.action))
        try:
            result = self.action(*self.args, **self.kwargs)
            self.result_ready.emit(result)
        except TaskCancelled as exc:
            log.info("Task %s cancelled: %s", name, exc)
            self.cancelled.emit(str(exc) or "Cancelled")
        except Exception as exc:
            log.exception("Task %s failed", name)
            self.error_occurred.emit(str(exc))


class TaskManager(QObject):
    def __init__(self, cfg=None):
        super().__init__()
        self.ui = cfg.ui if cfg is not None else UiConfig()
        self.worker = None
        self.overlay = None
        self.active_button = None
        self.active_live = False
        self._orig_text: dict = {}

    def _orig(self, button: QPushButton) -> str:
        if button not in self._orig_text:
            self._orig_text[button] = button.text()
        return self._orig_text[button]

    # ---- starting / stopping ---- #
    def run_task(self, button: QPushButton, label: QLabel, action, callback=None, live: bool = False,
                *args, **kwargs):
        orig = self._orig(button)

        def _set_status(text: str) -> None:
            label.setText(text)
            label.setToolTip(text)

        if self.worker and self.worker.isRunning():
            if button is self.active_button:
                self.worker.stop()
                _set_status(f"{'Stopping' if self.active_live else 'Cancelling'} {orig}...")
                log.info("Stop requested for %s", orig)
            else:
                self._show_busy(button, orig)
            return

        self.active_button, self.active_live = button, live
        tint = kwargs.pop("tint", QColor(*self.ui.overlay_color))
        self.overlay = ButtonProgressOverlay(button, label, color=tint, ui=self.ui)

        self.worker = UniversalWorker(action, *args, **kwargs)
        self.worker.progress_updated.connect(self.overlay.update_progress)
        self.worker.error_occurred.connect(lambda err: _set_status(f"Error: {err}"))
        self.worker.cancelled.connect(lambda msg: _set_status(msg))
        if callback:
            self.worker.result_ready.connect(callback)
        self.worker.finished.connect(self.cleanup)

        button.setText(f"{'Stop' if live else 'Cancel'} {orig}")
        button.setEnabled(True)          # stays clickable so it can be used to stop/cancel
        self.overlay.set_live_mode(live)

        log.info("Task started: %s (live=%s)", orig, live)
        self.worker.start()

    def cleanup(self) -> None:
        w, self.worker = self.worker, None
        if w:
            # w.wait() produces unwanted hang ups in the audio pipeline
            w.deleteLater()
        btn = self.active_button
        if self.overlay:
            self.overlay.deleteLater()
            self.overlay = None
        if btn:
            btn.setText(self._orig_text.get(btn, btn.text()))
            btn.setEnabled(True)
            log.info("Task finished: %s", self._orig_text.get(btn, btn.text()))
        self.active_button, self.active_live = None, False

    # ---- busy feedback ---- #
    def _show_busy(self, button: QPushButton, orig: str) -> None:
        active_orig = self._orig(self.active_button) if self.active_button else "another task"
        button.setText(f"Busy with {active_orig}")
        log.info("Busy: %s blocked while %s is running", orig, active_orig)
        QTimer.singleShot(self.ui.busy_message_ms, lambda: self._clear_busy(button, orig))

    @staticmethod
    def _clear_busy(button: QPushButton, orig: str) -> None:
        if button.text().startswith("Busy with"):
            button.setText(orig)


class ButtonProgressOverlay(QWidget):
    def __init__(self, parent_button: QPushButton, status_label: QLabel,
                color=QColor(119, 185, 0, 100), ui: UiConfig | None = None):
        super().__init__(parent_button)
        self.ui = ui or UiConfig()
        self.button = parent_button
        self.label = status_label
        self.base_color = QColor(color)
        self._alpha = self.base_color.alpha()
        self._display_percentage = 0.0

        self.animation = QVariantAnimation(self)
        self.animation.setDuration(self.ui.progress_anim_ms)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.valueChanged.connect(self._on_anim_value_changed)

        lo, hi = self.ui.pulse_alpha_range
        self.pulse_group = QSequentialAnimationGroup(self)
        self.pulse_up = QVariantAnimation()
        self.pulse_up.setDuration(self.ui.pulse_period_ms)
        self.pulse_up.setStartValue(lo)
        self.pulse_up.setEndValue(hi)
        self.pulse_up.setEasingCurve(QEasingCurve.Type.InOutSine)

        self.pulse_down = QVariantAnimation()
        self.pulse_down.setDuration(self.ui.pulse_period_ms)
        self.pulse_down.setStartValue(hi)
        self.pulse_down.setEndValue(lo)
        self.pulse_down.setEasingCurve(QEasingCurve.Type.InOutSine)

        self.pulse_group.addAnimation(self.pulse_up)
        self.pulse_group.addAnimation(self.pulse_down)
        self.pulse_group.setLoopCount(-1)
        self.pulse_up.valueChanged.connect(self._on_pulse_changed)
        self.pulse_down.valueChanged.connect(self._on_pulse_changed)

        self.is_live_mode = False

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.update_geometry()
        self.button.installEventFilter(self)
        self.hide()

    def _on_anim_value_changed(self, value):
        self._display_percentage = value
        self.update()

    def _on_pulse_changed(self, alpha):
        self._alpha = alpha
        self.update()

    def update_progress(self, current, total, status):
        if self.isHidden():
            self.show()
        if self.label and status:
            self.label.setText(status)
            self.label.setToolTip(status)

        if self.is_live_mode:
            # Realtime only ever sends status text (see bin/realtime.py's status_msg calls with
            # current=total=0): the bar stays fully lit and pulses, and is never re-animated from
            # here, so a stream of messages can't stutter the pulse.
            return

        target_pc = (current / total) if total > 0 else 0.0
        if abs(self._display_percentage - target_pc) < 0.001:
            return
        self.animation.stop()
        self.animation.setStartValue(self._display_percentage)
        self.animation.setEndValue(target_pc)
        self.animation.start()

    def set_live_mode(self, enabled: bool = True) -> None:
        self.is_live_mode = enabled
        if enabled:
            self._display_percentage = 1.0
            self.pulse_group.start()
            self.show()
        else:
            self.pulse_group.stop()
            self._alpha = self.base_color.alpha()
        self.update()

    def update_geometry(self) -> None:
        if self.button:
            self.setGeometry(self.button.rect())

    def eventFilter(self, a0, a1):
        if a0 == self.button and a1 is not None and a1.type() == QEvent.Type.Resize:
            self.update_geometry()
        return super().eventFilter(a0, a1)

    def paintEvent(self, a0):
        if not isinstance(self._display_percentage, (int, float)) or self._display_percentage <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        draw_color = QColor(self.base_color)
        draw_color.setAlpha(int(self._alpha))
        progress_width = int(self.width() * self._display_percentage)

        path = QPainterPath()
        path.addRoundedRect(0, 0, progress_width, self.height(), 4, 4)
        gradient = QLinearGradient(0, 0, progress_width, 0)
        gradient.setColorAt(0, draw_color)
        gradient.setColorAt(1, draw_color.lighter(125))

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.fillPath(path, QBrush(gradient))
        painter.end()