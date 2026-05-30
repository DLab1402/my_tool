import sys
import os
import json
import numpy as np
from scipy.signal import resample_poly, find_peaks, butter, filtfilt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel
)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
from template_gen import temp_find

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ======================
# Matplotlib Canvas  (dual-axis version for plot1)
# ======================

class MplCanvas(FigureCanvas):
    def __init__(self, dual_axis: bool = False):
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = None

        if dual_axis:
            # ECG on top, shares x-axis, independent y-axis
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel("ECG (norm)", color="darkorange", fontsize=8)
            self.ax2.tick_params(axis='y', labelcolor="darkorange")

        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)

        super().__init__(self.fig)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.updateGeometry()


# ======================
# Main Window
# ======================
class MainWindow(QWidget):
    cur_data    = None
    cur_sig     = None
    cur_ecg     = None          # downsampled + aligned ECG in PPG space
    cur_r_peaks = None          # R-peak positions in PPG space (after lag)
    cur_label   = None
    cur_peak    = None
    cur_temp    = None
    cur_index   = 0
    cur_valid   = None
    cur_file    = None

    def __init__(self, data_path):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        self.setWindowTitle("PPG / ECG Label Viewer")
        self.resize(1400, 750)
        self.cur_rect  = Rectangle((0, 0), 1, 1, edgecolor='red', alpha=0.3)
        self.data_path = data_path

        main_layout = QVBoxLayout(self)

        # -------- Plot 1 (Top Full Width) — dual-axis --------
        self.plot1 = MplCanvas(dual_axis=True)
        self.plot1.ax.set_ylim(-0.2, 1.2)
        if self.plot1.ax2:
            self.plot1.ax2.set_ylim(-0.2, 1.2)
        main_layout.addWidget(self.plot1, 1)

        # -------- Bottom Row --------
        bottom_row = QHBoxLayout()

        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(200)
        self.load_json_list()
        self.file_list.itemClicked.connect(self.load_json)
        bottom_row.addWidget(self.file_list, 1)

        self.plot2 = MplCanvas(dual_axis=False)
        self.plot2.ax.set_ylim(-0.2, 1.2)
        bottom_row.addWidget(self.plot2, 1)

        # Buttons
        button_layout = QVBoxLayout()
        self.back_btn      = QPushButton("Back")
        self.next_btn      = QPushButton("Next")
        self.error_btn     = QPushButton("Mark as error")
        self.undefined_btn = QPushButton("Mark as undefined")
        self.normal_btn    = QPushButton("Mark as normal")
        self.af_btn        = QPushButton("Mark as AF")
        self.pvc_btn       = QPushButton("Mark as PVC")
        self.pac_btn     = QPushButton("Mark as PAC")
        self.save_btn      = QPushButton("Save")

        button_layout.addWidget(self.back_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.error_btn)
        button_layout.addWidget(self.undefined_btn)
        button_layout.addWidget(self.normal_btn)
        button_layout.addWidget(self.af_btn)
        button_layout.addWidget(self.pvc_btn)
        button_layout.addWidget(self.pac_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()

        bottom_row.addLayout(button_layout, 1)
        main_layout.addLayout(bottom_row, 1)

        self.back_btn.clicked.connect(self.back_action)
        self.next_btn.clicked.connect(self.next_action)
        self.error_btn.clicked.connect(self.error_action)
        self.undefined_btn.clicked.connect(self.undefined_action)
        self.normal_btn.clicked.connect(self.normal_action)
        self.af_btn.clicked.connect(self.af_action)
        self.pvc_btn.clicked.connect(self.pvc_action)
        self.pac_btn.clicked.connect(self.pac_action)
        self.save_btn.clicked.connect(self.save_action)

    # ======================
    # Load JSON list
    # ======================
    def load_json_list(self):
        self.file_list.clear()
        files = [f for f in os.listdir(self.data_path) if f.endswith(".json")]
        self.file_list.addItems(files)

    # ======================
    # Load JSON file
    # ======================
    def load_json(self, item):
        self.plot1.ax.clear()
        if self.plot1.ax2:
            self.plot1.ax2.clear()
            self.plot1.ax2.set_ylim(-0.2, 1.2)

        self.cur_index = 0
        self.cur_file  = os.path.join(self.data_path, item.text())

        try:
            with open(self.cur_file, "r") as f:
                self.cur_data = json.load(f)

            self.cur_sig = self.cur_data.get("PPG", None)
            self.cur_ecg = self.cur_data.get("ECG", None)
            self.cur_label = self.cur_data.get("Label", None)

            self.cur_peak  = [[], []]
            self.cur_valid = []

            if "Test" in self.cur_data:
                self.cur_temp = self.cur_data["Test"]
                for seg in self.cur_data["Test"]:
                    self.cur_peak[0].append(seg["Pos"][0])
                    color = {
                                1: 'none',     # normal
                                0: 'red',      # error
                                -1: 'gray',    # undefined
                                2: 'purple',   # AF
                                3: 'orange',   # PVC
                                4: 'brown'     # PAC
                            }.get(seg["Valid"], 'gray')
                    rect  = Rectangle(
                        (seg["Pos"][0], 0),
                        seg["Pos"][1] - seg["Pos"][0], 1,
                        facecolor=color, edgecolor='none', alpha=0.3
                    )
                    self.cur_valid.append(rect)
                    self.plot1.ax.add_patch(rect)

                self.cur_peak[0].append(self.cur_data["Test"][-1]["Pos"][0])
                self.cur_peak[1] = [self.cur_sig[i] for i in self.cur_peak[0]]

                self.plot1.ax.text(
                    0.98, 0.98, "SAVED",
                    transform=self.plot1.ax.transAxes,
                    ha="right", va="top", fontsize=14, color="green"
                )
            else:
                templates = []
                _, peak, temp = temp_find(self.cur_sig).temping()
                for i in range(len(temp)):
                    templates.append({"Pos": [peak[i], peak[i + 1]], "Valid": 1, "Temp": temp[i]})
                    self.cur_peak[0].append(peak[i])
                    rect = Rectangle(
                        (templates[-1]["Pos"][0], 0),
                        templates[-1]["Pos"][1] - templates[-1]["Pos"][0], 1,
                        facecolor='none', edgecolor='none', alpha=0.3
                    )
                    self.cur_valid.append(rect)
                    self.plot1.ax.add_patch(rect)

                self.cur_data["Test"] = templates
                self.cur_peak[0].append(peak[-1])
                self.cur_peak[1] = [self.cur_sig[i] for i in self.cur_peak[0]]

            p1 = self.cur_data["Test"][self.cur_index]["Pos"][0]
            p2 = self.cur_data["Test"][self.cur_index]["Pos"][1]
            self.cur_rect.set_x(p1)
            self.cur_rect.set_width(p2 - p1)
            self.plot1_handle()
            self.plot1.ax.add_patch(self.cur_rect)
            self.plot2_handle(self.cur_data["Test"][0]["Temp"])

        except Exception as e:
            print("Error:", e)
            import traceback; traceback.print_exc()

    # ======================
    # Button Actions
    # ======================
    def back_action(self):
        if self.cur_index > 0:
            self.cur_index -= 1
            self._move_rect_to(self.cur_index)

    def next_action(self):
        if self.cur_data and self.cur_index < len(self.cur_data["Test"]) - 1:
            self.cur_index += 1
            self._move_rect_to(self.cur_index)

    def _move_rect_to(self, idx):
        p1 = self.cur_data["Test"][idx]["Pos"][0]
        p2 = self.cur_data["Test"][idx]["Pos"][1]
        self.cur_rect.set_x(p1)
        self.cur_rect.set_width(p2 - p1)
        self.plot2_handle(self.cur_data["Test"][idx]["Temp"])
        self.plot1.draw()

    def error_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("red")
            self.cur_data["Test"][self.cur_index]["Valid"] = 0
            self.plot1.draw()

    def undefined_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("gray")
            self.cur_data["Test"][self.cur_index]["Valid"] = -1
            self.plot1.draw()

    def normal_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("none")
            self.cur_data["Test"][self.cur_index]["Valid"] = 1
            self.plot1.draw()

    def af_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("purple")
            self.cur_data["Test"][self.cur_index]["Valid"] = 2
            self.plot1.draw()

    def pvc_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("orange")
            self.cur_data["Test"][self.cur_index]["Valid"] = 3
            self.plot1.draw()

    def pac_action(self):
        if self.cur_valid:
            self.cur_valid[self.cur_index].set_facecolor("brown")
            self.cur_data["Test"][self.cur_index]["Valid"] = 4
            self.plot1.draw()

    def save_action(self):
        def clean_numpy(obj):
            if isinstance(obj, dict):   return {k: clean_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [clean_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):  return obj.tolist()
            elif isinstance(obj, np.integer):  return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            else: return obj

        self.cur_data = clean_numpy(self.cur_data)
        with open(self.cur_file, "w") as f:
            json.dump(self.cur_data, f, indent=4)

    # ======================
    # Plot functions
    # ======================
    def plot1_handle(self):
        ax  = self.plot1.ax
        ax2 = self.plot1.ax2
        n   = len(self.cur_sig)
        x   = np.arange(n)

        # --- PPG (primary axis, blue) ---
        ax.set_xlim(0, n)
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel("PPG (norm)", color="steelblue", fontsize=8)
        ax.tick_params(axis='y', labelcolor="steelblue")
        ax.plot(x, np.array(self.cur_sig), color="steelblue", linewidth=1, label="PPG")
        ax.plot(self.cur_peak[0], self.cur_peak[1],
                marker='o', linestyle='None',
                color='green', markerfacecolor='green', markersize=5, label="PPG peaks")

        # Label is optional
        if self.cur_label is not None:
            label_arr = np.array(self.cur_label)
            if len(label_arr) == n:          # sanity-check length matches PPG
                ax.plot(x, label_arr, color="red", linewidth=0.8, alpha=0.7, label="Label")

        # --- ECG (secondary axis, orange) — cur_ecg is always shape (ppg_len,) ---
        if ax2 is not None and self.cur_ecg is not None:
            assert len(self.cur_ecg) == n, \
                f"ECG length {len(self.cur_ecg)} != PPG length {n}"  # should never fire
            ax2.set_xlim(0, n)
            ax2.set_ylim(-0.2, 1.2)
            ax2.plot(x, self.cur_ecg, color="darkorange", linewidth=0.8,
                     alpha=0.75, label="ECG (aligned)")

            # R-peaks markers on ECG axis
            if self.cur_r_peaks is not None and len(self.cur_r_peaks) > 0:
                rp = self.cur_r_peaks
                ax2.plot(rp, self.cur_ecg[rp],
                         marker='v', linestyle='None',
                         color='darkred', markersize=5, label="R-peaks")

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc="upper left", fontsize=7, framealpha=0.5)

        self.plot1.draw()

    def plot2_handle(self, temp):
        self.plot2.ax.clear()
        self.plot2.ax.plot(np.array(temp))
        self.plot2.draw()

    # ======================
    # Keyboard shortcuts
    # ======================
    def keyPressEvent(self, event):
        key = event.key()
        if   key == Qt.Key.Key_Right: self.next_action()
        elif key == Qt.Key.Key_Left:  self.back_action()
        elif key == Qt.Key.Key_S:     self.save_action()
        elif key == Qt.Key.Key_Q:     self.error_action()
        elif key == Qt.Key.Key_W:     self.undefined_action()
        elif key == Qt.Key.Key_E:     self.normal_action()
        elif key == Qt.Key.Key_R:     self.af_action()
        elif key == Qt.Key.Key_T:     self.pvc_action()
        elif key == Qt.Key.Key_Y:     self.pac_action()
        else: super().keyPressEvent(event)


# ======================
# Run
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow("H:\\My Drive\\data_set_review\\data_notem")
    window.show()
    sys.exit(app.exec())