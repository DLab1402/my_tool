import json
import numpy as np
import matplotlib.pyplot as plt

file = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\propose_model\epoch.json"
with open(file, "r") as f:
    epoch = json.load(f)
    train = np.array(epoch["train"])
    val = np.array(epoch["val"])
    x = np.array(range(len(train)))
    plt.plot(np.arange(1, 51),train,label="Training loss")
    plt.plot(np.arange(1, 51),val,label="Validation loss")
    plt.grid()
    plt.legend()
    plt.xlim((1,50))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()