import sys
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ======================
# Your Matplotlib Canvas
# ======================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.updateGeometry()

    def plot_example(self):
        # Example data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        self.ax.clear()

        # Plot green markers
        self.ax.plot(x, y, 'o', color='green', markersize=5)

        # Shade red area between curve and zero line
        self.ax.fill_between(x, y, 0, color='red', alpha=0.3)

        # Set ranges
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-1.5, 1.5)

        self.draw()

# ======================
# Main Window
# ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Matplotlib in PyQt Example")
        self.setGeometry(100, 100, 800, 500)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        self.setCentralWidget(central_widget)

        # Draw the plot
        self.canvas.plot_example()

# ======================
# Run App
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
