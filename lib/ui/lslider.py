from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider

class LiveParams:
    """Shared realtime parameter container accessible across threads/loops."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

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

class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(object)

    def __init__(self, title, min_val, max_val, default_val, suffix="", step=1, scale=1.0, on_change=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.scale = scale
        self.suffix = suffix
        self.on_change = on_change

        self.lblTitle = QLabel(title)
        self.lblValue = QLabel()

        self.sldMain = ClickSlider(Qt.Orientation.Horizontal)
        self.sldMain.setRange(min_val, max_val)
        self.sldMain.setSingleStep(step)
        self.sldMain.setPageStep(step * 2)

        layMain = QVBoxLayout(self)
        layTop = QHBoxLayout()

        layTop.addWidget(self.lblTitle)
        layTop.addStretch()
        layTop.addWidget(self.lblValue)

        layMain.addLayout(layTop)
        layMain.addWidget(self.sldMain)

        self.sldMain.valueChanged.connect(self.handle_value_changed)
        
        # default_val is expected to be in UI space (e.g., 100 for 100%)
        self.setValue(default_val)

    def handle_value_changed(self, value):
        # Snapping offset calculated relative to min_val
        snapped_value = round((value - self.min_val) / self.step) * self.step + self.min_val
        snapped_value = max(self.min_val, min(self.max_val, snapped_value))

        self.sldMain.blockSignals(True)
        self.sldMain.setValue(int(snapped_value))
        self.sldMain.blockSignals(False)

        # 1. The label displays the UI integer (e.g., 100%)
        self.lblValue.setText(f"{snapped_value:g}{self.suffix}")

        # 2. The backend gets the mathematically scaled value (e.g., 100 * 0.01 = 1.0)
        backend_val = snapped_value * self.scale

        if self.on_change:
            self.on_change(backend_val)
        self.valueChanged.emit(backend_val)

    def value(self):
        # Return the mathematically scaled value (e.g., 1.0)
        return self.sldMain.value() * self.scale

    def setValue(self, val):
        # val is assumed to be in UI integer space (e.g., 100)
        raw_val = int(val)
        self.sldMain.setValue(raw_val)
        self.handle_value_changed(raw_val)