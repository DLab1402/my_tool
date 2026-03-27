import json
import os
import matplotlib.pyplot as plt
import random
import numpy as np

# Load all JSON files from a folder
data = []
valid = 0
error = 0
aryhmias = [0]*14
template = []

path = "D:/my_project/valid_data_8_2_26_resample2"

path = "D:/my_project/valid_data_8_2_26_resample"
files = [f for f in os.listdir(path) if f.endswith(".json")]

n = len(files)
idx = random.randint(0, n - 1)
print("idx:", idx)
idx = 86

file_path = os.path.join(path, files[idx])

with open(file_path, "r") as f:
    data = json.load(f)
    label = np.array(data["Syn_Label"])
    
    ecg = np.array(data["Syn_ECG"])
    ecg = ecg[500:3000]
    ppg = np.array(data["Syn_PPG"])
    ppg = ppg[500:3000]
    ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))+1
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    plt.plot(ppg[500:3000],label = "PPG")
    plt.plot(ecg[500:3000],label = "ECG")
    plt.legend()
    # plt.plot(label[500:3000])
    plt.grid()
    plt.show()
    

    fig, ax = plt.subplots(4, 8, figsize=(15, 6))
    ax = ax.flatten()

    template = data["Template"]
    item = random.sample(range(len(template)), 32)
    
    for i,id in enumerate(item):
        ax[i].plot(template[id]["Temp"])
        ax[i].grid()
    
    plt.show()



#86: normal
#131: arhythmia
#95: AF