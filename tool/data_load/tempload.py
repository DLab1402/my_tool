import os
import json
import torch

class TempDataset(torch.utils.data.Dataset):

    def __init__(self, root_folder):

        self.signals = []
        self.labels = []  # keep if you still want labels later

        # list all json files in folder
        json_files = [f for f in os.listdir(root_folder) if f.endswith(".json")]
        print(f"Found {len(json_files)} json files.")

        for fname in json_files:
            path = os.path.join(root_folder, fname)

            with open(path, "r") as file:
                data = json.load(file)   # <-- this is a DICT

            # read the "Template" field (list of dicts)
            templates = data.get("Template", [])

            for item in templates:
                # each item is a dict with keys: "Temp" and "Valid"
                if item.get("Valid", 0) == 1:
                    temp_signal = item.get("Temp")
                    self.signals.append(temp_signal)
                    self.labels.append(1)  # dummy label for now

    def __getitem__(self, index):
        signal = torch.tensor(self.signals[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.signals)