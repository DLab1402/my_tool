import json
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

INFERENT_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\inferent.json"

with open(INFERENT_PATH, "r") as f:
    data = json.load(f)
    input = np.array(data["raw_template"])
    reconstruct = np.array(data["reconstruct"])
    # disease = np.array(data["disease"])
    loss = np.array(data["loss"])
    pred = np.array(data["predict"])
    lab = np.array(data["label"])

ind = np.random.choice(len(input), size=24, replace=False)

fig, ax = plt.subplots(4, 6, figsize=(20, 16))
ax = ax.flatten()

for i, idx in enumerate(ind):
    ax[i].plot(input[idx], label="Input", color="blue")
    ax[i].plot(reconstruct[idx], label="Reconstruct", color="orange")
    # ax[i].set_title(f"Sample {disease[idx]}) (Pred: {pred[idx]}, Label: {lab[idx]})", fontsize=10)
    ax[i].set_title(f"(Pred: {pred[idx]}, Label: {lab[idx]}, Loss: {loss[idx]:.4f})", fontsize=10)
    ax[i].text(50, 0.8, f"Skewness: {skew(input[idx]):.4f}\nLoss: {loss[idx]:.4f}", fontsize=10, color='black')
    ax[i].grid()

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()