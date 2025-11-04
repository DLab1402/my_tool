import os
import json
import shutil

# Declaration of source and destination folders
src_folder = r"D:\ppg_project\Data\labeled"
dst_folder = r"D:\ppg_project\Data\valid_data"

# Create the destination folder if not exists
os.makedirs(dst_folder, exist_ok=True)

# Iterate through all files in source folder
for filename in os.listdir(src_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(src_folder, filename)
    
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if "Valid" is exist and it is True
            if "Valid" in data:
                if data["Valid"] == True:            
                    shutil.copy(file_path, dst_folder)
                    print(f" Copied: {filename}")
                else:
                    print(f" Skipped (no Valid): {filename}")
        
        except Exception as e:
            print(f" Error reading {filename}: {e}")
