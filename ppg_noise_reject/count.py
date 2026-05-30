import os
import json

DATA_PATH = r"H:\My Drive\data_set_review\data_run"

def count_samples(file_path):
    file_list = os.listdir(file_path)
    error_count = 0
    normal_count = 0
    af_count = 0
    pac_pvc_count = 0
    for filename in file_list:
        if filename.endswith(".json"):
            with open(os.path.join(file_path, filename), "r") as f:
                data = json.load(f)
                for seg in data["Test"]:
                    if seg["Valid"] == 0:
                        error_count += 1
                    elif seg["Valid"] == 1:
                        normal_count += 1
                    elif seg["Valid"] == 2:
                        af_count += 1
                    elif seg["Valid"] in [3, 4]:
                        pac_pvc_count += 1 
    print(f"Total set: {error_count + normal_count + af_count + pac_pvc_count}")
    print(f"  Error: {error_count}")
    print(f"  Normal: {normal_count}")
    print(f"  AF: {af_count}")
    print(f"  PAC/PVC: {pac_pvc_count}")

print("Training set:")
count_samples(os.path.join(DATA_PATH,"train"))
print("\nValidation set:")
count_samples(os.path.join(DATA_PATH,"validate"))
print("\nTest set:")
count_samples(os.path.join(DATA_PATH,"test"))
