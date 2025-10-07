import os
import json

# Thư mục chứa các file json
src_folder = r"D:\ppg_project\Data\valid_data"

lengths = []  # lưu số điểm của từng file
lengths_ecg = []
lengths_label = []
for filename in os.listdir(src_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(src_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "Syn_PPG" in data:
                length = len(data["Syn_PPG"])
                lengths.append(length)
                print(f"{filename}: {length} points")
            # if "Syn_ECG" in data:
            #     length_ECG = len(data["Syn_ECG"])
            #     lengths_ecg.append(length_ECG)
            #     print(f"{filename}: {length_ECG} points")
            # if "Syn_Label" in data:
            #     length_label = len(data["Syn_Label"])
            #     lengths_label.append(length_label)
            #     print(f"{filename}: {length_label} points")
            else:
                print(f"{filename}: NO given keys ")
        
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

# --- Thống kê ---
if lengths:
    avg_len = sum(lengths) / len(lengths)
    print("\n📊 Statistics:")
    print(f"  - Total files: {len(lengths)}")
    print(f"  - Min length : {min(lengths)}")
    print(f"  - Max length : {max(lengths)}")
    print(f"  - Avg length : {avg_len:.2f}")
