import os
import json
import shutil
import random

# ======= CHANGE THESE =======
src_folder = r"H:\My Drive\data_set_review\data_notem"
out_root = r"H:\My Drive\data_set_review\data_run"

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25
# ============================

# create output folders
for split in ["train", "validate", "test"]:
    os.makedirs(os.path.join(out_root, split), exist_ok=True)

# collect only json files that contain "Template"
valid_files = []

for fname in os.listdir(src_folder):
    if not fname.endswith(".json"):
        continue

    full_path = os.path.join(src_folder, fname)

    try:
        with open(full_path, "r") as f:
            data = json.load(f)

        # check if it has "Test"
        if isinstance(data, dict) and "Test" in data:
            valid_files.append(fname)

    except Exception as e:
        print(f"Skipping {fname} (error reading): {e}")

print(f"Total valid JSON with 'Test': {len(valid_files)}")

# shuffle before splitting
random.shuffle(valid_files)

n = len(valid_files)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)

train_files = valid_files[:n_train]
val_files   = valid_files[n_train:n_train+n_val]
test_files  = valid_files[n_train+n_val:]

# move files
for fname in train_files:
    shutil.copy(os.path.join(src_folder, fname),
                os.path.join(out_root, "train", fname))

for fname in val_files:
    shutil.copy(os.path.join(src_folder, fname),
                os.path.join(out_root, "validate", fname))

for fname in test_files:
    shutil.copy(os.path.join(src_folder, fname),
                os.path.join(out_root, "test", fname))

print("Done!")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")