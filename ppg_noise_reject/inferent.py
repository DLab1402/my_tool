import json
import numpy as np
import matplotlib.pyplot as plt

TEST_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\propose_model\depict.json"
with open(TEST_PATH, "r") as f:
    data = json.load(f)
    lab = data["label"]
    ppg = data["ppg"]
    adjust = data["adjust"]
    for i in [5,4]:
        plt.figure(figsize=(20, 4))

        sig = np.array(ppg[i][500:8500])
        la = np.array(lab[i][500:8500])
        ad = np.array(adjust[i][500:8500])
    
        x = np.arange(0, len(sig))

        # Prediction (gray)
        plt.plot(x,sig,color="#2980b9", linewidth=2, alpha=0.8, label="Prediction")
        
        # Ground truth (blue)
        # plt.plot(x,la,color="#2980b9", linewidth=2.5, label="Ground Truth")

        # Adjust as shaded area (red transparent band)
        plt.fill_between(x,ad,0,where=(ad > 0),color="#e74c3c", alpha=0.2, label="Adjustment Area")
        
        # Titles and labels
        # plt.title(f"PPG Signal Comparison with Adjustment Area (Sample {i})",fontsize=16, fontweight='bold')
        plt.xlabel("Samples", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)

        # Grid
        # plt.grid(True, linestyle='--', alpha=0.4)

        # # Clean style
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)

        # plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.show()