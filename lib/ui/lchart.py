from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QObject, pyqtSignal


class ChartBridge(QObject):
    """Training runs in a worker thread; widgets must only be touched from the GUI thread.
    Emitting a signal from the worker is delivered to the chart on the GUI thread."""
    cleared = pyqtSignal()
    updated = pyqtSignal(int, float, float, float, float, float, float)

    def __init__(self, chart):
        super().__init__(chart)
        self.updated.connect(chart.update_plot)
        self.cleared.connect(chart.clear)

    def clear(self):
        """Thread-safe call to clear the chart."""
        self.cleared.emit()

    def update_plot(self, epoch, g, d, mel, fm, kl, val):
        self.updated.emit(int(epoch), float(g), float(d), float(mel), float(fm), float(kl), float(val))


class LossChart(QWidget):
    sample = pyqtSignal(int, float, float, float, float, float, float)
    cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        # 1. Main GraphicsLayoutWidget so we can stack plot + legend vertically
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(None)
        self.glw.setStyleSheet("background-color: transparent;")

        # 2. Add Plot in row 0
        self.wdgPlot = self.glw.addPlot(row=0, col=0, title="GAN Training Progress (RVC)")
        self.wdgPlot.showGrid(x=True, y=True, alpha=0.3)

        # 3. Create Horizontal Legend in row 1
        # horiz=True arranges items horizontally; colCount=5 puts all 5 items in 1 line
        self.legend = pg.LegendItem(colCount=6, horiz=True, offset=(0, 0))
        self.legend.setParentItem(self.glw.ci)  # Attach to canvas item container
        self.glw.nextRow()
        self.glw.addItem(self.legend)

        # 4. Plot lines and add them to the legend
        self.gen_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#5AC684", width=2), name="Generator (G)"
        )
        self.disc_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#ff5555", width=2), name="Discriminator (D)"
        )
        self.mel_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#953ddd", width=2), name="Melodic (Mel)"
        )
        self.fm_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#2e7adf", width=2), name="Feature (FM)"
        )
        self.kl_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#d8e439", width=2), name="Divergence (KL)"
        )
        self.val_line = self.wdgPlot.plot(
            pen=pg.mkPen(color="#7b3f11", width=2), name="Validation x20 (Val)"
        )

        self.legend.addItem(self.gen_line, name="Generator (G)")
        self.legend.addItem(self.disc_line, name="Discriminator (D)")
        self.legend.addItem(self.mel_line, name="Melodic (Mel)")
        self.legend.addItem(self.fm_line, name="Feature (FM)")
        self.legend.addItem(self.kl_line, name="Divergence (KL)")
        self.legend.addItem(self.val_line, name="Validation x20 (Val)")

        layMain = QVBoxLayout(self)
        layMain.setContentsMargins(0, 0, 0, 0)
        layMain.addWidget(self.glw)

        self.x_data = []
        self.gen_y = []
        self.disc_y = []
        self.mel_y = []
        self.fm_y = []
        self.kl_y = []
        self.val_y = []

        self.sample.connect(self.update_plot)
        self.cleared.connect(self.clear)

    def clear(self):
        self.x_data.clear()
        self.gen_y.clear()
        self.disc_y.clear()
        self.mel_y.clear()
        self.fm_y.clear()
        self.kl_y.clear()
        self.val_y.clear()

        self.gen_line.clear()
        self.disc_line.clear()
        self.mel_line.clear()
        self.fm_line.clear()
        self.kl_line.clear()
        self.val_line.clear()

        self.wdgPlot.enableAutoRange(axis="xy", enable=True)

    def update_plot(self, epoch, g_loss, d_loss, mel_loss, fm_loss, kl_loss, val_loss):
        self.x_data.append(epoch)
        self.gen_y.append(g_loss)
        self.disc_y.append(d_loss)
        self.mel_y.append(mel_loss)
        self.fm_y.append(fm_loss)
        self.kl_y.append(kl_loss)
        self.val_y.append(val_loss)

        self.gen_line.setData(self.x_data, self.gen_y)
        self.disc_line.setData(self.x_data, self.disc_y)
        self.mel_line.setData(self.x_data, self.mel_y)
        self.fm_line.setData(self.x_data, self.fm_y)
        self.kl_line.setData(self.x_data, self.kl_y)
        self.val_line.setData(self.x_data, self.val_y)

        window_size = 50
        if len(self.x_data) > 0:
            latest_x = self.x_data[-1]
            if latest_x > window_size:
                self.wdgPlot.setXRange(latest_x - window_size, latest_x)
            else:
                self.wdgPlot.setXRange(0, window_size)

        self.wdgPlot.enableAutoRange(axis="y", enable=True)