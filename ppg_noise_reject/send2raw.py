import os
import json

PATH = "H:\\My Drive\\data_set_review\\data"
SAVE_PATH = "H:\\My Drive\\data_set_review\\data_notem"
file_list = [f for f in os.listdir(PATH) if f.endswith(".json")]

for file in file_list:
    file_path = os.path.join(PATH, file)
    with open(file_path, 'r') as f:
        data = json.load(f)
    ppg = data["PPG"]
    ecg = data["ECG"]
    label = data["Label"]
    raw_data = {
        "PPG": ppg,
        "ECG": ecg,
        "Label": label
    }
    raw_file_path = os.path.join(SAVE_PATH, file)
    with open(raw_file_path, 'w') as f:
        json.dump(raw_data, f)