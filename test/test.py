import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


# ==========================================
# SIGNAL VIEWER CLASS
# ==========================================

class SignalViewer:
    def __init__(self, fig,
                 top_pos=[0.10, 0.55, 0.85, 0.35],
                 bottom_pos=[0.10, 0.20, 0.65, 0.25],
                 window_size=300):

        self.fig = fig
        self.window_size = window_size
        self.index = 0
        self.signal = None

        self.ax_top = fig.add_axes(top_pos)
        self.ax_bottom = fig.add_axes(bottom_pos)

    def set_signal(self, signal_array):
        self.signal = np.array(signal_array)
        self.index = 0
        self.update()

    def next_window(self, step=100):
        if self.signal is None:
            return
        self.index = min(self.index + step,
                         len(self.signal) - self.window_size)
        self.update()

    def prev_window(self, step=100):
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

        # Highlight zoom area
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


# ==========================================
# CREATE EXAMPLE DATA (AUTO)
# ==========================================

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

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

# ==========================================
# MAIN FIGURE
# ==========================================

fig = plt.figure(figsize=(12, 7))

viewer = SignalViewer(fig)

# ===== Buttons =====
ax_back = fig.add_axes([0.80, 0.28, 0.12, 0.07])
ax_next = fig.add_axes([0.80, 0.20, 0.12, 0.07])

btn_back = Button(ax_back, "Back")
btn_next = Button(ax_next, "Next")

# ===== File Selection Slider =====
ax_slider = fig.add_axes([0.10, 0.08, 0.80, 0.05])

file_slider = Slider(
    ax=ax_slider,
    label="File Index",
    valmin=0,
    valmax=len(json_files) - 1,
    valinit=0,
    valstep=1
)


# ==========================================
# FUNCTIONS
# ==========================================

def load_file_by_index(idx):
    filename = json_files[int(idx)]
    path = os.path.join(DATA_FOLDER, filename)

    with open(path, "r") as f:
        data = json.load(f)

    viewer.set_signal(data["signal"])
    fig.suptitle(f"Current File: {filename}", fontsize=12)


def slider_update(val):
    load_file_by_index(val)


def next_event(event):
    viewer.next_window()


def back_event(event):
    viewer.prev_window()


# Connect events
file_slider.on_changed(slider_update)
btn_next.on_clicked(next_event)
btn_back.on_clicked(back_event)

# Load first file
load_file_by_index(0)

plt.show()
