import json
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def check_similarity_anderson(data1, data2):
    # Thực hiện kiểm định Anderson-Darling cho 2 mẫu
    result = stats.anderson_ksamp([data1, data2])
    
    print(f"Anderson-Darling statistic: {result.statistic:.4f}")
    
    # Các ngưỡng giá trị tới hạn (Critical values) tương ứng với các mức ý nghĩa
    # [25%, 10%, 5%, 2.5%, 1%, 0.5%, 0.1%]
    critical_values = result.critical_values
    sig_levels = [25, 10, 5, 2.5, 1, 0.5, 0.1]
    
    print("-" * 30)
    print("Mức ý nghĩa (%) | Giá trị tới hạn")
    for lvl, cv in zip(sig_levels, critical_values):
        print(f"{lvl:14}% | {cv:.4f}")
    print("-" * 30)

    # Giải thích kết quả
    # Nếu statistic < critical_value ở mức 1% (0.01), ta KHÔNG bác bỏ giả thuyết H0
    # Nghĩa là 2 phân phối GIỐNG NHAU với độ tin cậy 99%
    alpha_1_percent_cv = critical_values[4] 
    
    if result.statistic < alpha_1_percent_cv:
        print("Kết luận: Hai phân phối GIỐNG NHAU (Độ tin cậy 99%)")
    else:
        print("Kết luận: Hai phân phối CÓ SỰ KHÁC BIỆT (Độ tin cậy 99%)")

    return result

# Chạy thử nghiệm
# res = check_similarity_anderson(data1, data2)

with open("H:\\My Drive\\data_set_ppg_reject4\\model\\latent.json", "r") as f:
    data = json.load(f)
    latent = np.array(data["latent"])
    print(latent.shape)
    idx0 = np.where(np.array(data["disease"]) == 0)[0]
    idx1 = np.where(np.array(data["disease"]) == 1)[0]
    idx2 = np.where(np.array(data["disease"]) > 1)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent[idx0, 0], latent[idx0, 1], latent[idx0, 2], label = "Healthy")
    ax.scatter(latent[idx1, 0], latent[idx1, 1], latent[idx1, 2], label = "AF")
    ax.scatter(latent[idx2, 0], latent[idx2, 1], latent[idx2, 2], label = "PAC/PVC")
    plt.legend()
    plt.show()

with open("H:\\My Drive\\data_set_ppg_reject4\\model\\loss.json", "r") as f:
    loss = json.load(f)
    train = np.array(loss["train"])
    val = np.array(loss["val"])

    stat, p = ks_2samp(train, val)

    print("KS statistic:", stat)
    print("p-value:", p)

    wd = wasserstein_distance(train, val)

    print("Wasserstein Distance:", wd)

    result = check_similarity_anderson(train, val)
    print(result)

    plt.hist(train, bins = np.linspace(0, 0.0004, 100), label = "Training loss")
    plt.hist(val, bins = np.linspace(0, 0.0004, 100), label = "Validation loss")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()

    plt.show()

with open("H:\\My Drive\\data_set_ppg_reject4\\model\\auc.json", "r") as f:
    auc = json.load(f)
    fpr = np.array(auc["fpr"])
    thr = np.array(auc["thr"])

    

    plt.plot(fpr, thr, label = "Training loss")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()

    plt.show()