import sys
import os
import json
import numpy as np

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
# Matplotlib Canvas
# ======================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 🔥 remove big margins
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)

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
    cur_data = None
    cur_sig = None
    cur_label = None
    cur_peak = None
    cur_temp = None
    cur_index = 0
    cur_valid = None
    cur_file = None
    def __init__(self,data_path):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        self.setWindowTitle("Correct Layout Viewer")
        self.resize(1400, 700)
        self.cur_rect = Rectangle((0, 0), 1, 1, edgecolor='red', alpha=0.3)
        self.data_path = data_path

        main_layout = QVBoxLayout(self)

        # -------- Plot 1 (Top Full Width) --------
        self.plot1 = MplCanvas()
        self.plot1.ax.set_ylim(-0.2, 1.2)
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
        self.plot2.ax.set_ylim(-0.2, 1.2)
        # self.plot2.ax.set_xlim(0, 100)
        bottom_row.addWidget(self.plot2, 1)

        # Buttons
        button_layout = QVBoxLayout()
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        self.error_btn = QPushButton("Mark as error")
        self.undefined_btn = QPushButton("Mark as undefined")
        self.valid_btn = QPushButton("Mark as valid")
        self.save_btn = QPushButton("Save")

        button_layout.addWidget(self.back_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.error_btn)
        button_layout.addWidget(self.undefined_btn)
        button_layout.addWidget(self.valid_btn)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()

        bottom_row.addLayout(button_layout, 1)

        main_layout.addLayout(bottom_row, 1)

        self.back_btn.clicked.connect(self.back_action)
        self.next_btn.clicked.connect(self.next_action)  
        self.error_btn.clicked.connect(self.error_action)
        self.undefined_btn.clicked.connect(self.undefined_action)
        self.valid_btn.clicked.connect(self.undefined_action)
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
        self.cur_index = 0
        self.cur_file = os.path.join(self.data_path, item.text())
        
        try:
            with open(self.cur_file, "r") as f:
                self.cur_data = json.load(f)

            min_val = min(self.cur_data["Syn_PPG"])
            max_val = max(self.cur_data["Syn_PPG"])

            self.cur_sig = [(x - min_val)/(max_val - min_val) for x in self.cur_data["Syn_PPG"]]
            self.cur_label = [1 if x > 1 else x for x in self.cur_data["Syn_Label"]]
            self.cur_peak = [[],[]]
            self.cur_valid = []
            if "Template" in self.cur_data:
                self.cur_temp = self.cur_data["Template"]
                for item in self.cur_data["Template"]:
                    self.cur_peak[0].append(item["Pos"][0])
                    if item["Valid"] == 1:
                        color = 'none'
                    elif item["Valid"] == 0:
                        color = 'red'
                    else:
                        color = 'gray'
                    valid_rec = Rectangle((item["Pos"][0], 0), item["Pos"][1]-item["Pos"][0], 1, facecolor= color, edgecolor='none', alpha=0.3)
                    self.cur_valid.append(valid_rec)
                    self.plot1.ax.add_patch(valid_rec)

                self.cur_peak[0].append(self.cur_data["Template"][-1]["Pos"][0])
                self.cur_peak[1] = [self.cur_sig[i] for i in self.cur_peak[0]]

                self.plot1.ax.text(
                    0.98, 0.98,
                    "SAVED",
                    transform=self.plot1.ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=14,
                    color="green")

            else:
                templates = []
                _,peak,temp = temp_find(self.cur_sig).temping()
                for i in range(len(temp)):
                    templates.append({"Pos": [peak[i],peak[i+1]], "Valid": 1, "Temp": temp[i]})
                    self.cur_peak[0].append(peak[i])
                    valid_rec = Rectangle((templates[-1]["Pos"][0], 0), templates[-1]["Pos"][1]-templates[-1]["Pos"][0], 1, facecolor='none', edgecolor='none', alpha=0.3)
                    self.cur_valid.append(valid_rec)
                    self.plot1.ax.add_patch(valid_rec)

                self.cur_data["Template"] = templates
                self.cur_peak[0].append(peak[-1])
                self.cur_peak[1] = [self.cur_sig[i] for i in self.cur_peak[0]]

            p1 = self.cur_data["Template"][self.cur_index]["Pos"][0]
            p2 = self.cur_data["Template"][self.cur_index]["Pos"][1]
            self.cur_rect.set_x(p1)
            self.cur_rect.set_width(p2-p1)
            self.plot1_handle()
            self.plot1.ax.add_patch(self.cur_rect)
            self.plot2_handle(self.cur_data["Template"][0]["Temp"])

        except Exception as e:
            print("Error:", e)

    # ======================
    # Button Actions
    # ======================

    def back_action(self):
        if self.cur_index > 0:
            self.cur_index -= 1
            p1 = self.cur_data["Template"][self.cur_index]["Pos"][0]
            p2 = self.cur_data["Template"][self.cur_index]["Pos"][1]
            self.cur_rect.set_x(p1)
            self.cur_rect.set_width(p2-p1)
            self.plot2_handle(self.cur_data["Template"][self.cur_index]["Temp"])
            self.plot1.draw()

    def next_action(self):
        if self.cur_index < len(self.cur_data["Template"]) - 1:
            self.cur_index += 1
            p1 = self.cur_data["Template"][self.cur_index]["Pos"][0]
            p2 = self.cur_data["Template"][self.cur_index]["Pos"][1]
            self.cur_rect.set_x(p1)
            self.cur_rect.set_width(p2-p1)
            self.plot2_handle(self.cur_data["Template"][self.cur_index]["Temp"])
            self.plot1.draw()

    def error_action(self):
        self.cur_valid[self.cur_index].set_facecolor("red")
        self.cur_data["Template"][self.cur_index]["Valid"] = 0


    def undefined_action(self):
        self.cur_valid[self.cur_index].set_facecolor("gray")
        self.cur_data["Template"][self.cur_index]["Valid"] = -1

    def valid_action(self):
        self.cur_valid[self.cur_index].set_facecolor("none")
        self.cur_data["Template"][self.cur_index]["Valid"] = 1

    def save_action(self):
        def clean_numpy(obj):
            if isinstance(obj, dict):
                return {k: clean_numpy(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                return [clean_numpy(v) for v in obj]

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, np.integer):
                return int(obj)

            elif isinstance(obj, np.floating):
                return float(obj)

            else:
                return obj
        
        self.cur_data = clean_numpy(self.cur_data)
        with open(self.cur_file, "w") as f:
            json.dump(self.cur_data, f, indent=4)

    # ======================
    # Plot functions
    # ======================
    def plot1_handle(self):
        self.plot1.ax.set_xlim(0,len(self.cur_sig))
        self.plot1.ax.plot(np.array(self.cur_sig))
        self.plot1.ax.plot(self.cur_peak[0], self.cur_peak[1], 
             marker='o', 
             linestyle='None',   
             color='green',     
             markerfacecolor='green')  

        self.plot1.ax.plot(np.array(self.cur_label), color = "red")
        self.plot1.draw()

    def plot2_handle(self,temp):
        self.plot2.ax.clear()
        self.plot2.ax.plot(np.array(temp))
        self.plot2.draw()   
        
    # ======================
    # Plot functions
    # ======================
    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Right:
            self.next_action()

        elif event.key() == Qt.Key.Key_Left:
            self.back_action()

        elif event.key() == Qt.Key.Key_S:
            self.save_action()      # you create this

        elif event.key() == Qt.Key.Key_Q:
            self.error_action()
        
        elif event.key() == Qt.Key.Key_W:
            self.undefined_action()

        elif event.key() == Qt.Key.Key_E:
            self.valid_action()

        else:
            super().keyPressEvent(event)

# ======================
# Run
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow("D:/my_project/valid_data_8_2_26_resample2")
    # window = MainWindow("H:\\My Drive\\data_set_ppg_reject2\\train")
    window.show()
    sys.exit(app.exec())