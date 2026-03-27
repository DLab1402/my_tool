import os
import json
import numpy as np

out_root = "H:/My Drive/data_set_ppg_reject4/"

def counter(path, type="other"):
# collect only json files that contain "Template"
    normal = 0
    arhythmia = 0
    tem_nor = 0
    tem_af = 0
    tem_p = 0

    for fname in os.listdir(path):
        if not fname.endswith(".json"):
            continue
        full_path = os.path.join(path, fname)

        try:
            with open(full_path, "r") as f:
                data = json.load(f)

            # check if it has "Template"
            if type == "test":
                if "Test" not in data:
                    continue
            if isinstance(data, dict) and "Test" in data:
                if "Label" in data:
                    if np.sum(data["Label"]) == 0:
                        normal += 1
                    else:
                        arhythmia += 1

                for item in data["Test"]:
                    if item.get("Valid", 0) != 0:
                        p1,p2 = item.get("Pos", [])
                        pos = item.get("Pos", [0, 0])
                        p1, p2 = pos
                        if np.max(data.get("Label", [])[p1:p2]) == 0:
                            tem_nor += 1
                        elif np.max(data.get("Label", [])[p1:p2]) == 1:
                            tem_af += 1
                        else:
                            tem_p += 1

        except Exception as e:
            print(f"Skipping {fname} (error reading): {e}")

    print(f"Total normal: {normal}")
    print(f"Total arhythmia: {arhythmia}")
    print(f"Total tem_nor: {tem_nor}")
    print(f"Total tem_af: {tem_af}")
    print(f"Total tem_p: {tem_p}")

if __name__ == "__main__":
    # print("Counting in train set:")
    # train_path = os.path.join(out_root, "train")
    # counter(train_path) 
    # print("\nCounting in validate set:")
    # val_path = os.path.join(out_root, "validate")
    # counter(val_path)
    print("\nCounting in test set:")
    test_path = os.path.join(out_root, "test")
    counter(test_path,"test")