import json
import matplotlib.pyplot as plt

# Đường dẫn file JSON
file_path = r"D:\ppg_project\Data\valid_data_resampled\12_20230424140640_wave.csv_chunk0.json"

# Đọc file JSON
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy dữ liệu
ppg = data.get("Syn_PPG", [])
ecg = data.get("Syn_ECG", [])
label = data.get("Syn_Label", [])

# Vẽ chart
plt.figure(figsize=(12, 6))

if ppg:
    plt.plot(ppg, label="Syn_PPG")
if ecg:
    plt.plot(ecg, label="Syn_ECG")
if label:
    plt.plot(label, label="Syn_Label")

plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title("PPG/ECG/Label signals")
plt.legend()
plt.grid(True)
plt.show()
