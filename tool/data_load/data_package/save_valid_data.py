import os
import json
import shutil

# Thư mục chứa file json gốc
src_folder = r"D:\ppg_project\Data\labeled"
# Thư mục đích để copy file hợp lệ
dst_folder = r"D:\ppg_project\Data\valid_data"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(dst_folder, exist_ok=True)

# Duyệt qua tất cả file trong src_folder
for filename in os.listdir(src_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(src_folder, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Kiểm tra nếu file có key "Valid"
            if "Valid" in data:
                dst_path = os.path.join(dst_folder, filename)
                if not os.path.exitsts(dst_path):
                    shutil.copy(file_path, dst_folder)
                    print(f"✅ Copied: {filename}")
                else:
                    print(f"❌ Skipped (no Valid): {filename}")
        
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
