from template_gen import temp_find
from scipy.signal import resample, savgol_filter
from torch.utils.data import Dataset,DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import torch

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder):

        self.signals = []
        self.labels = []
        self.target_length = 128
        self.a = temp_find()

        json_files = [f for f in os.listdir(root_folder) if f.endswith(".json")]
        print(f"Found {len(json_files)} json files.")

        for fname in json_files:
            path = os.path.join(root_folder, fname)

            with open(path, "r") as file:
                data = json.load(file)

            templates = data.get("Template", [])
            s = data.get("Syn_PPG",[])

            for item in templates:
                if item.get("Valid", 0) == 1:
                    p1,p2 = item.get("Pos", [])
                    temp = np.array(item.get("Temp"))
                    # temp = np.array(s[p1:p2])
                    # temp = self.a.liesample(temp,128)
                    y_smooth = savgol_filter(temp, window_length=11, polyorder=3)
                    temp = resample(y_smooth, self.target_length)
                    min_val = temp.min()
                    max_val = temp.max()
                    final_signal = (temp - min_val) / (max_val - min_val + 1e-8)
                    self.signals.append(final_signal)
                    self.labels.append(1)

    def __getitem__(self, index):
        signal = self.signals[index]
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.signals)

class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.signals = []
        self.labels = []
        self.target_length = 128
        self.a = temp_find()

        json_files = [f for f in os.listdir(root_folder) if f.endswith(".json")]
        print(f"Found {len(json_files)} json files.")

        for fname in json_files:
            path = os.path.join(root_folder, fname)

            with open(path, "r") as file:
                data = json.load(file)

            templates = data.get("Template", [])
            s = data.get("Syn_PPG",[])

            for item in templates:
                  if item.get("Valid", 0) == 1:
                    p1,p2 = item.get("Pos", [])
                    temp = np.array(item.get("Temp",[]))
                    # temp = np.array(s[p1:p2])
                    # temp = self.a.liesample(temp,128)
                    y_smooth = savgol_filter(temp, window_length=11, polyorder=3)
                    temp = resample(y_smooth, self.target_length)
                    min_val = temp.min()
                    max_val = temp.max()
                    final_signal = (temp - min_val) / (max_val - min_val + 1e-8)

                    self.signals.append(final_signal)
                    label = [0, 0]
                    label[0] = item.get("Valid", 0)

                    pos = item.get("Pos", [0, 0])
                    p1, p2 = pos
                    label[1] = np.max(data.get("Label", [])[p1:p2])

                    self.labels.append(label)

    def __getitem__(self, index):
        signal = self.signals[index]
        signal = self.signals[index]
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return signal, label

    def __len__(self):
      return len(self.signals)

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.signals = []
        self.labels = []
        self.target_length = 128
        self.a = temp_find()

        json_files = [f for f in os.listdir(root_folder) if f.endswith(".json")]
        print(f"Found {len(json_files)} json files.")

        for fname in json_files:
            path = os.path.join(root_folder, fname)

            with open(path, "r") as file:
                data = json.load(file)

            templates = data.get("Test", [])
            s = data.get("PPG",[])

            for item in templates:
                        p1,p2 = item.get("Pos", [])
                        temp = np.array(item.get("Temp",[]))
                        # temp = np.array(s[p1:p2])
                        # temp = self.a.liesample(temp,128)
                        y_smooth = savgol_filter(temp, window_length=11, polyorder=3)
                        temp = resample(y_smooth, self.target_length)
                        min_val = temp.min()
                        max_val = temp.max()
                        final_signal = (temp - min_val) / (max_val - min_val + 1e-8)

                        self.signals.append(final_signal)
                        label = [0, 0]
                        label[0] = item.get("Valid", 0)

                        pos = item.get("Pos", [0, 0])
                        p1, p2 = pos
                        label[1] = np.max(data.get("Label", [])[p1:p2])

                        self.labels.append(label)

    def __getitem__(self, index):
        signal = self.signals[index]
        signal = self.signals[index]
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return signal, label

    def __len__(self):
      return len(self.signals)
