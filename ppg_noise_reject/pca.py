import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TRAIN_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\train.json"
VAL_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\validate.json"
TEST_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\inferent.json"

with open(TRAIN_PATH, "r") as f:
    data = json.load(f)
    latent_train = np.array(data["latent"])
    disease_train = np.array(data["disease"])
    latent_train = latent_train.reshape(latent_train.shape[0], -1)

with open(VAL_PATH, "r") as f:
    data = json.load(f)
    latent_val = np.array(data["latent"])
    disease_val = np.array(data["disease"])
    latent_val = latent_val.reshape(latent_val.shape[0], -1)

with open(TEST_PATH, "r") as f:
    data = json.load(f)
    latent_test = np.array(data["latent"])
    disease_test = np.array(data["disease"])
    latent_test = latent_test.reshape(latent_test.shape[0], -1)

def pca_analysis(latent,disease, n_components=3):
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent)
    latent_pca = pca.transform(latent)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance (2D):", np.sum(pca.explained_variance_ratio_))

    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    idx0 = np.where(disease == 1)[0]
    idx1 = np.where(disease >= 2)[0]
    # idx2 = np.where(disease > 2)[0]
    ax1.scatter(latent_pca[idx0, 0], latent_pca[idx0, 1], label = "Healthy")
    ax1.scatter(latent_pca[idx1, 0], latent_pca[idx1, 1], label = "Arrhythmia")
    # ax1.scatter(latent_pca[idx2, 0], latent_pca[idx2, 1], label = "PAC/PVC")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Test set")
    ax1.grid()
    ax1.legend()
    ax2.scatter(latent_pca[idx0, 0], latent_pca[idx0, 1], latent_pca[idx0, 2], label = "Healthy")
    ax2.scatter(latent_pca[idx1, 0], latent_pca[idx1, 1], latent_pca[idx1, 2], label = "Arrhythmia")
    # ax2.scatter(latent_pca[idx2, 0], latent_pca[idx2, 1], latent_pca[idx2, 2], label = "PAC/PVC")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()
    ax2.grid()
    plt.show()

pca_analysis(latent_test,disease_test)