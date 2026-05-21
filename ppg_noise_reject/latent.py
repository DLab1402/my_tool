import json
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\propose_model\train.json"
VAL_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\propose_model\validate.json"
TEST_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\propose_model\inferent.json"

with open(TRAIN_PATH, "r") as f:
    data = json.load(f)
    train_latent = np.array(data["latent"])
    test_label = np.array(data["label"])

with open(VAL_PATH, "r") as f:
    data = json.load(f)
    val_latent = np.array(data["latent"])
    test_label = np.array(data["label"])

with open(TEST_PATH, "r") as f:
    data = json.load(f)
    test_latent = np.array(data["latent"])
    test_label = np.array(data["label"])
    label = np.array(data["label"])

ind = np.random.choice(1867, size=16, replace=False)

def latent_sketch(latent, ind):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    for i, idx in enumerate(ind):
        if latent.ndim < 4:
            axes[i].plot(latent[idx][0])
            axes[i].set_title(f"Sample {idx} (Label: {test_label[idx]})") 
        else:
            axes[i].imshow(latent[idx][0],aspect='auto',cmap='gray_r',interpolation='nearest')
            axes[i].set_title(f"Sample {idx} (Label: {test_label[idx]})")   
    plt.show()

def latent_heatmap(train_latent, val_latent, test_latent):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes = axes.flatten()

    print("Train latent shape:", train_latent.shape)
    if train_latent.ndim == 4:
        train_latent = train_latent.squeeze(1)
        train_latent = train_latent.reshape(train_latent.shape[0], train_latent.shape[2]*train_latent.shape[3])
        train_latent = train_latent.T
        val_latent = val_latent.squeeze(1)
        val_latent = val_latent.reshape(val_latent.shape[0], val_latent.shape[2]*val_latent.shape[3])
        val_latent = val_latent.T
        test_latent = test_latent.squeeze(1)
        test_latent = test_latent.reshape(test_latent.shape[0], test_latent.shape[2]*test_latent.shape[3])
        test_latent = test_latent.T
    else:
        train_latent = train_latent.squeeze().T
        val_latent = val_latent.squeeze().T
        test_latent = test_latent.squeeze().T

    axes[0].imshow(train_latent, aspect='auto', cmap='gray_r', interpolation='nearest')
    axes[0].set_title("Training Latent Heatmap")
    axes[0].set_ylabel("Hidden layer")
    axes[1].imshow(val_latent, aspect='auto', cmap='gray_r', interpolation='nearest')
    axes[1].set_title("Validation Latent Heatmap")
    axes[1].set_ylabel("Hidden layer")
    axes[2].imshow(test_latent[:,np.where(label == 1)[0]], aspect='auto', cmap='gray_r', interpolation='nearest')
    axes[2].set_title("Test Latent Heatmap")
    axes[2].set_ylabel("Hidden layer")
    axes[2].set_xlabel("Samples")


    plt.show()

# latent_sketch(test_latent, ind)
latent_heatmap(train_latent, val_latent, test_latent)