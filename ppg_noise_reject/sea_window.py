import json
import numpy as np
import matplotlib.pyplot as plt

T_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\depict.json"

pred = []
lab = []
seg = 480

input = []
reconstruct = []

ratio = 0 # Define ratio here, as it was used but not defined

with open(T_PATH, "r") as f:
    data = json.load(f)
    label = data["label"]
    ppg = data["ppg"]
    adjust = data["adjust"]
    for i in range(len(ppg)):
      current_signal_length = len(ppg[i])
      # Ensure num_sections is at least 1 to avoid ValueError if signal is shorter than seg
      num_sections = max(1, current_signal_length // seg)

      ppg_segs = np.array_split(ppg[i], num_sections)

      current_label_length = len(label[i])
      num_label_sections = max(1, current_label_length // seg)
      label_segs = np.array_split(label[i], num_label_sections)

      current_adjust_length = len(adjust[i])
      num_adjust_sections = max(1, current_adjust_length // seg)
      predict_segs = np.array_split(adjust[i], num_adjust_sections)

      for j in range(len(ppg_segs)):
        # Ensure we don't divide by zero if a segment is empty
        len_label_seg = len(label_segs[j])
        R = np.sum(label_segs[j]) / (len_label_seg if len_label_seg > 0 else 1)
        if R > ratio:
          L = 0
        else:
          L = 1
        lab.append(L)

        len_predict_seg = len(predict_segs[j])
        pred_R = np.sum(predict_segs[j]) / (len_predict_seg if len_predict_seg > 0 else 1)
        if pred_R > ratio:
          pred_L = 0
        else:
          pred_L = 1
        pred.append(pred_L)
        input.append(ppg_segs[j])
        reconstruct.append(adjust[i][j*seg:(j+1)*seg])

# Convert pred and lab to numpy arrays for boolean indexing
pred = np.array(pred)
lab = np.array(lab)

TP = np.sum((pred == 1) & (lab == 1))
FP = np.sum((pred == 1) & (lab == 0))
FN = np.sum((pred == 0) & (lab == 1))
TN = np.sum((pred == 0) & (lab == 0))



print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)

# Add a small epsilon to the denominator to prevent division by zero for accuracy, precision, recall, f1
total_sum = (TP + TN + FP + FN)
accuracy = (TP + TN) / (total_sum if total_sum > 0 else 1)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


ind = np.random.choice(len(input), size=18, replace=False)

fig, ax = plt.subplots(6, 3, figsize=(20, 16))
ax = ax.flatten()

for i, idx in enumerate(ind):
    ax[i].plot(input[idx], label="Input", color="blue")
    ax[i].plot(reconstruct[idx], label="Reconstruct", color="orange")
    # ax[i].set_title(f"Sample {disease[idx]}) (Pred: {pred[idx]}, Label: {lab[idx]})", fontsize=10)
    # ax[i].set_title(f"(Pred: {pred[idx]}, Label: {lab[idx]}, Loss: {loss[idx]:.4f})", fontsize=10)
    # ax[i].text(50, 0.8, f"Skewness: {skew(input[idx]):.4f}\nLoss: {loss[idx]:.4f}", fontsize=10, color='black')
    ax[i].grid()

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()