import os
import json
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

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

        TPR.append(TP / P)
        FPR.append(FP / N)

    return np.array(FPR), np.array(TPR), thresholds

def sqi(path, path_test):
    files = [f for f in os.listdir(path) if f.endswith(".json")]

    skew_train = []
    temp_train = []
    dis_train = []
    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        if "Template" in data:
            template = data["Template"]
            for temp in template:
                if temp["Valid"]:
                    skew_train.append(skew(temp["Temp"]))
                    temp_train.append(temp["Temp"])
                    p1,p2 = temp["Pos"]
                    n = np.max(data["Syn_Label"][p1:p2])
                    dis_train.append(n)       

    skew_train = np.array(skew_train)
    dis_train = np.array(dis_train)
    id0 = np.where(dis_train == 0)[0]
    id1 = np.where(dis_train != 0)[0]


    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax.flatten()
    ax[0].hist(skew_train, bins=20)
    ax[1].hist(skew_train[id0], bins=20)
    ax[2].hist(skew_train[id1], bins=20)

    plt.show()

    min = np.min(skew_train)
    max = np.max(skew_train)

    files = [f for f in os.listdir(path_test) if f.endswith(".json")]

    skew_test = []
    temp_test = []
    preds = []
    label = []

    for file in files:
        file_path = os.path.join(path_test, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        if "Test" in data:
            template = data["Test"]
            for temp in template:
                skew_test.append(skew(temp["Temp"]))
                temp_test.append(temp["Temp"])
                label.append(temp["Valid"])

    skew_test = np.array(skew_test)

    for i in range(len(skew_test)):
        if skew_test[i] < min or skew_test[i] > max:
            preds.append(0)
        else:
            preds.append(1)

    preds = np.array(preds)
    label = np.array(label)

    TP = ((preds == 1) & (label == 1)).sum().item()
    TN = ((preds == 0) & (label == 0)).sum().item()
    FP = ((preds == 1) & (label == 0)).sum().item()
    FN = ((preds == 0) & (label == 1)).sum().item()

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

    fpr, tpr, thr = roc_curve_manual(label, skew_test)

    auc = compute_auc(fpr, tpr)


    print("AUC:", auc)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    path = "H:/My Drive/data_set_ppg_reject3/train"
    path_test = "H:/My Drive/data_set_ppg_reject3/test"
    sqi(path, path_test)