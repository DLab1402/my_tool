import os
import json

TRAIN_PATH = r"H:\My Drive\data_set_ppg_reject4 - Copy\train"
VAL_PATH = r"H:\My Drive\data_set_ppg_reject4 - Copy\validate"
TEST_PATH = r"H:\My Drive\data_set_ppg_reject4 - Copy\test"
def reform_training(path):
    with open(path, "r") as f:
        data = json.load(f)
        data["label"] = 1
    PPG = data["Syn_PPG"]
    ECG = data["Syn_ECG"]
    LABEL = data["Syn_Label"]
    temp = data["Template"]
    VALID = []
    POS = []
    for tem in temp:
        VALID.append(tem["Valid"])
        POS.append(tem["Pos"])
    
    return {"PPG": PPG,"ECG": ECG,"VALID": VALID,"POS": POS, "LABEL": LABEL}

def reform_testing(path):
    with open(path, "r") as f:
        data = json.load(f)
        data["label"] = 1
    PPG = data["PPG"]
    ECG = data["ECG"]
    LABEL = data["Label"]
    temp = data["Test"]
    VALID = []
    POS = []
    for tem in temp:
        VALID.append(tem["Valid"])
        POS.append(tem["Pos"])
    
    return {"PPG": PPG,"ECG": ECG,"VALID": VALID,"POS": POS, "LABEL": LABEL}

for file_list in os.listdir(TRAIN_PATH):
    print(f"Processing {file_list}...")
    data = reform_training(os.path.join(TRAIN_PATH, file_list))
    with open(os.path.join(r"H:\My Drive\data_set_ppg_reject4 - Copy\train_new", file_list), "w") as f:        
        json.dump(data, f)

for file_list in os.listdir(TEST_PATH):
    try:
        print(f"Processing {file_list}...")
        data = reform_testing(os.path.join(TEST_PATH, file_list))
        with open(os.path.join(r"H:\My Drive\data_set_ppg_reject4 - Copy\test_new", file_list), "w") as f:        
            json.dump(data, f)
    except Exception as e:
        print(f"Error processing {file_list}: {e}")

for file_list in os.listdir(VAL_PATH):
    print(f"Processing {file_list}...")
    data = reform_training(os.path.join(VAL_PATH, file_list))
    with open(os.path.join(r"H:\My Drive\data_set_ppg_reject4 - Copy\validate_new", file_list), "w") as f:        
        json.dump(data, f)