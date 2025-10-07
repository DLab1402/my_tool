import os
import json

# Thư mục chứa file json gốc
src_folder = r"D:\ppg_project\Data\valid_data"
# Thư mục để lưu các file json đã cắt
dst_folder = r"D:\ppg_project\Data\valid_data_resampled"
os.makedirs(dst_folder, exist_ok=True)

chunk_size = 320

for filename in os.listdir(src_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(src_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Đảm bảo đủ cả 3 key
            if all(k in data for k in ["Syn_PPG", "Syn_ECG", "Syn_Label"]):
                ppg = data["Syn_PPG"]
                ecg = data["Syn_ECG"]
                label = data["Syn_Label"]

                # Tính số chunk nguyên vẹn
                num_chunks = min(len(ppg), len(ecg), len(label)) // chunk_size

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = start + chunk_size

                    chunk_data = {
                        "Syn_PPG": ppg[start:end],
                        "Syn_ECG": ecg[start:end],
                        "Syn_Label": label[start:end]
                    }

                    # Tạo tên file mới
                    base, ext = os.path.splitext(filename)
                    out_filename = f"{base}_chunk{i}{ext}"
                    out_path = os.path.join(dst_folder, out_filename)

                    with open(out_path, "w", encoding="utf-8") as f_out:
                        json.dump(chunk_data, f_out)

                print(f"✅ {filename}: split into {num_chunks} chunks (discarded leftover)")
            else:
                print(f"❌ {filename} missing one of Syn_PPG / Syn_ECG / Syn_Label")

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
