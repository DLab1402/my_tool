import json
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

INFERENT_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\inferent.json"
DISEASE_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\disease.json"

with open(INFERENT_PATH, "r") as f:
    data = json.load(f)
    pred = np.array(data["predict"])
    lab = np.array(data["label"])
    loss = np.array(data["loss"])

def compute_auc(fpr, tpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc

def roc_curve_manual(y_true, y_score):
    thresholds = np.sort(np.unique(y_score))
    TPR = []
    FPR = []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    for t in thresholds:
        y_pred = y_score <= t
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        TPR.append(TP / P if P else 0)
        FPR.append(FP / N if N else 0)
    return np.array(FPR), np.array(TPR), thresholds

fpr, tpr, thr = roc_curve_manual(lab, loss)

TP = np.sum((pred == 1) & (lab == 1))
FP = np.sum((pred == 1) & (lab == 0))
FN = np.sum((pred == 0) & (lab == 1))
TN = np.sum((pred == 0) & (lab == 0))



print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

auc = compute_auc(fpr, tpr)

print("AUC:", auc)
J = tpr - fpr
ix = np.argmax(J)

best_thresh = thr[ix]
print("Best Threshold:", best_thresh)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()