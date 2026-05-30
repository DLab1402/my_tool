import json
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

TRAIN_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\train.json"
VAL_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\validate.json"
TEST_PATH = r"G:\My Drive\Target_in_2026\D.Lab\my_publish\ppg_denoise\review\same_kernel\inferent.json"

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


def calculate_wasserstein_distance(data1, data2, label1="Distribution 1", label2="Distribution 2"):
    """
    Calculate Wasserstein distance between two distributions
    
    Args:
        data1: First dataset (list or array)
        data2: Second dataset (list or array)
        label1: Name of first distribution
        label2: Name of second distribution
    
    Returns:
        float: Wasserstein distance
    """
    wd = wasserstein_distance(data1, data2)
    print(f"Wasserstein distance ({label1} vs {label2}): {wd}")
    return wd

with open(TRAIN_PATH, "r") as f:
    data = json.load(f)
    loss_train = data["loss"]

with open(VAL_PATH, "r") as f:
    data = json.load(f)
    loss_val = data["loss"]

with open(TEST_PATH, "r") as f:
    data = json.load(f)
    loss_test = data["loss"]

print(check_similarity_anderson(loss_train, loss_val))
calculate_wasserstein_distance(loss_train, loss_val, "Training Loss", "Validation Loss")

plt.hist(loss_train, bins=500, alpha=0.5, label="Training Loss")
plt.hist(loss_val, bins=500, alpha=0.5, label="Validation Loss")
# plt.hist(loss_test, bins=50, alpha=0.5, label="Test Loss")
plt.xlim(0, 0.0005)
plt.xlabel("Loss Value")
plt.ylabel("Frequency")
plt.title("Distribution of Loss Values")
plt.grid()
plt.legend()
plt.show()


