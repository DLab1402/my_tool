import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from template_gen import temp_find

# ==========================================
# SIGNAL VIEWER CLASS
# ==========================================

class label_tool:
    def __init__(self, data_path, fig = None,
                 top_pos=[0.30, 0.55, 0.65, 0.35],
                 bottom_pos=[0.30, 0.15, 0.50, 0.30],
                 window_size=300):

        self.data_path = data_path
        self.data_list = sorted([f for f in os.listdir(data_path)
                        if f.endswith(".json")])
        
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        
        self.fig = fig
        self.window_size = window_size
        self.index = 0
        self.signal = None

        # Create axes
        self.ax_top = fig.add_axes(top_pos)
        self.ax_bottom = fig.add_axes(bottom_pos)
        
        # Left file list
        self.ax_tree = self.fig.add_axes([0.05, 0.10, 0.20, 0.80])
        self.ax_tree.set_xticks([])
        self.ax_tree.set_yticks([])
        self.ax_tree.set_title("Data Files")

        # Buttons
        ax_back = fig.add_axes([0.83, 0.30, 0.10, 0.08])
        ax_next = fig.add_axes([0.83, 0.20, 0.10, 0.08])

        btn_back = Button(ax_back, "Back")
        btn_next = Button(ax_next, "Next")

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        btn_next.on_clicked(self.next_event)
        btn_back.on_clicked(self.back_event)

        self.draw_file_list()
        
    def draw_file_list(self):
        self.ax_tree.clear()
        self.ax_tree.set_xticks([])
        self.ax_tree.set_yticks([])
        self.ax_tree.set_title("Data")

        for i, filename in enumerate(self.data_list):
            y = 0.95 - i * 0.07
            self.ax_tree.text(0.05, y, filename,
                        transform=self.ax_tree.transAxes,
                        verticalalignment='top',
                        picker=True)

        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        filename = event.artist.get_text()
        self.load_json(filename)

    def load_json(self, filename):
        path = os.path.join(self.data_path, filename)
        with open(path, "r") as f:
            data = json.load(f)

        self.set_signal(data["SynPPG"])

    def set_signal(self, signal_array):
        self.signal = np.array(signal_array)
        self.index = 0
        self.update()

    def next_event(self):
        if self.signal is None:
            return
        self.index = min(self.index + step,
                         len(self.signal) - self.window_size)
        self.update()


    def back_event(self):
        if self.signal is None:
            return
        self.index = max(0, self.index - step)
        self.update()

    def update(self):
        if self.signal is None:
            return

        self.ax_top.clear()
        self.ax_bottom.clear()

        x = np.arange(len(self.signal))

        # Full signal
        self.ax_top.plot(x, self.signal)
        self.ax_top.set_title("Full Signal")

        # Highlight zoom region
        self.ax_top.axvspan(self.index,
                            self.index + self.window_size,
                            alpha=0.2)

        # Zoomed region
        end = min(self.index + self.window_size,
                  len(self.signal))

        self.ax_bottom.plot(x[self.index:end],
                            self.signal[self.index:end])
        self.ax_bottom.set_title("Zoom Window")

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    DATA_FOLDER = "data"
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Auto-create example files if empty
    if len(os.listdir(DATA_FOLDER)) == 0:
        x = np.linspace(0, 20, 2000)
        signals = {
            "sine.json": np.sin(x),
            "cosine.json": np.cos(x),
            "ppg_like.json": np.sin(x) + 0.3*np.sin(3*x)
        }
        for name, sig in signals.items():
            with open(os.path.join(DATA_FOLDER, name), "w") as f:
                json.dump({"signal": sig.tolist()}, f)

    json_files = sorted([f for f in os.listdir(DATA_FOLDER)
                        if f.endswith(".json")])
    label_tool()
    plt.show()