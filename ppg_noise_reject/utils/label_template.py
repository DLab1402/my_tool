import sys
import os
import json
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel
)

from PyQt6.QtWidgets import QSizePolicy
from template_gen import temp_find

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ======================
# Matplotlib Canvas
# ======================
class MplCanvas(FigureCanvas):
    __cur_data = None
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # 🔥 Tell Qt this widget wants all available space
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.updateGeometry()
# ======================
# Main Window
# ======================
class MainWindow(QWidget):
    cur_data = None
    def __init__(self,data_path):
        super().__init__()
        self.setWindowTitle("Correct Layout Viewer")
        self.resize(1000, 700)

        self.data_path = data_path

        main_layout = QVBoxLayout(self)

        # -------- Plot 1 (Top Full Width) --------
        self.plot1 = MplCanvas()
        main_layout.addWidget(self.plot1, 1)

        # -------- Bottom Row (List + Plot2 + Buttons) --------
        bottom_row = QHBoxLayout()

        # JSON List
        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(200)
        self.load_json_list()
        self.file_list.itemClicked.connect(self.load_json)
        bottom_row.addWidget(self.file_list, 1)

        # Plot 2
        self.plot2 = MplCanvas()
        bottom_row.addWidget(self.plot2, 1)

        # Buttons
        button_layout = QVBoxLayout()
        self.back_btn = QPushButton("Back")
        self.mid_btn = QPushButton("Button")
        self.next_btn = QPushButton("Next")

        button_layout.addWidget(self.back_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.mid_btn)
        
        button_layout.addStretch()

        bottom_row.addLayout(button_layout, 1)

        main_layout.addLayout(bottom_row, 1)

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
        file_path = os.path.join(self.data_path, item.text())
        
        try:
            with open(file_path, "r") as f:
                self.cur_data = json.load(f)

            min_val = min(self.cur_data["Syn_PPG"])
            max_val = max(self.cur_data["Syn_PPG"])

            self.cur_sig = [(x - min_val)/(max_val - min_val) for x in self.cur_data["Syn_PPG"]]
            self.cur_label = [1 if x > 1 else x for x in self.cur_data["Syn_Label"]]
            self.cur_peak = [[],[]]
            if "Template" in self.cur_data:
                self.cur_temp = self.cur_data["Template"]
                for item in self.cur_data["Template"]:
                    self.cur_peak[0].append(item["Pos"][0])
                self.cur_peak[0].append(self.cur_data["Template"][-1]["Pos"][0])
                self.cur_peak[1] = self.cur_sig[self.cur_peak[0]]

            else:
                templates = []
                peak,temp = temp_find(self.cur_sig).temping()
                print(temp)
                for i in range(len(temp)):
                    templates.append({"Pos": [peak[i],peak[i+1]], "Valid": None, "Temp": temp[i]})

                self.cur_data["Template"] = templates

            self.plot1_handle()

        except Exception as e:
            print("Error:", e)

    # ======================
    # Example initial plot
    # ======================
    def plot1_handle(self):
        x = np.linspace(0, len(self.cur_sig))
        self.plot1.ax.clear()
        self.plot1.ax.plot(np.array(self.cur_sig))
        self.plot1.ax.plot(np.array(self.cur_label), color = "red")
        # self.plot2.ax.plot(np.cos())

        # self.plot1.draw()
        self.plot1.draw()


# ======================
# Run
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow("D:/ppg_project/Data/valid_data_8_2_26")
    window.show()
    sys.exit(app.exec())