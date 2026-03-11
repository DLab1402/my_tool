import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load all JSON files from a folder
data = []
valid = 0
error = 0
aryhmias = [0]*14
template = []
fig, ax = plt.subplots(4, 4, figsize=(15, 6))
ax = ax.flatten()
path = "D:/my_project/valid_data_8_2_26_resample"
for filename in os.listdir(path):
    if filename.endswith('.json'):
        with open(os.path.join(path, filename)) as f:
            data = json.load(f)
            label = data["Syn_Label"]
            if "Template" in data:
                template = data["Template"]
                for item in template:
                    temp = item["Temp"]
                    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                    if item["Valid"] == 1:
                        valid += 1
                        p1,p2 = item["Pos"]
                        l = np.max(label[p1:p2])
                        ax[0].plot(temp)
                        if l == 0:
                            aryhmias[0] += 1
                            ax[1].plot(temp)
                        elif l == 1:
                            aryhmias[1] += 1
                            ax[2].plot(temp)   
                        elif l == 2:
                            aryhmias[2] += 1
                            ax[3].plot(temp)
                        elif l == 3:
                            aryhmias[3] += 1
                            ax[4].plot(temp)
                        elif l == 4:
                            aryhmias[4] += 1
                            ax[5].plot(temp)
                        elif l == 5:
                            aryhmias[5] += 1
                            ax[6].plot(temp)
                        elif l == 6:
                            aryhmias[6] += 1
                            ax[7].plot(temp)
                        elif l == 7:
                            aryhmias[7] += 1
                            ax[8].plot(temp)
                        elif l == 8:
                            aryhmias[8] += 1
                            ax[9].plot(temp)
                        elif l == 9:
                            aryhmias[9] += 1
                            ax[10].plot(temp)
                        elif l == 10:
                            aryhmias[10] += 1
                            ax[11].plot(temp)
                        elif l == 11:
                            aryhmias[11] += 1
                            ax[12].plot(temp)
                        elif l == 12:
                            aryhmias[12] += 1
                            ax[13].plot(temp)
                        elif l == 13:
                            aryhmias[13] += 1
                            ax[14].plot(temp)
                    elif item["Valid"] == 0:
                        error += 1
                        ax[15].plot(temp)

print("valid:", valid)
print("error:", error)
print("aryhmias:", aryhmias)
plt.show()