import sys
sys.path.append('D:\my_project\my_tool\model')
from conv_1D import CNN1D
from MLP import MLP
import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from scipy.signal import resample, savgol_filter
from torch.utils.data import Dataset,DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

from template_gen import temp_find

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
                    self.labels.append([1,0])

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
            if True:
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
                      label[1] = np.max(data.get("Syn_Label", [])[p1:p2])

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

            if "Test" in data:
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
                          if item.get("Valid", 0) == 1:
                            label[0] = 1
                          else:
                            label[0] = 0

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

class CNNAuto(nn.Module):

    def __init__(self, encoder, decoder, hidden1 = None, hidden2 = None):
        super().__init__()
        self.encoder = CNN1D(encoder)
        self.decoder = CNN1D(decoder)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.vis = []

        if hidden1 is not None and hidden2 is not None:
          if len(hidden1["dim"]) > 1:
            self.first = hidden1["dim"][0][0]
            self.last  = hidden2["dim"][-1][1]
            self.mlp1 = MLP(hidden1)
            self.mlp2 = MLP(hidden2)
          else:
            self.first = self.last = hidden1["dim"][0]

          self.dim1 = None
          self.dim2 = None

        self.hidden = None

    def forward(self, x):
        self.vis.clear()
        encoded = self.encoder(x)   # (B, C, L)
        self.vis = self.encoder.vis

        if self.hidden1 is not None and self.hidden2 is not None:
          B, C, L = encoded.shape
          flat_dim = C * L

          encoded = torch.flatten(encoded, 1)
          self.vis.append(encoded)

          if self.dim1 is None:
            self.dim1 = nn.Linear(flat_dim, self.first)
            self.dim2 = nn.Linear(self.last, flat_dim)

          encoded = self.dim1(encoded)
          encoded = self.hidden1["rest"](encoded)
          self.vis.append(encoded)
          if len(self.hidden1["dim"])>1:
            encoded = self.mlp1(encoded)
            self.vis.extend(self.mlp1.vis)
          self.hidden = encoded
          if len(self.hidden1["dim"])>1:
            decoded = self.mlp2(encoded)
            self.vis.extend(self.mlp2.vis)
          else:
            decoded = encoded
          decoded = self.dim2(decoded)
          decoded = self.hidden2["rest"](decoded)
          self.vis.append(decoded)
          decoded = decoded.reshape(B, C, L)
          self.vis.append(decoded)
          decoded = self.decoder(decoded)

        else:
          self.hidden = encoded
          decoded = self.decoder(encoded)

        self.vis.extend(self.decoder.vis)

        return decoded

def create_model():
    return CNNAuto(
        encoder = {"type": ["conv", "conv","conv","conv"],
          "dim": [(1, 32),(32, 64),(64,128),(128,256)],
          "kernel": [(7,1,3),(5,1,2),(5,1,2),(3,1,1)],
          "pkernel": [(3, 2, 1),(3, 2, 1),(3, 2, 1),(3, 2, 1)],
          "pooling": [nn.MaxPool1d,nn.MaxPool1d,nn.MaxPool1d,nn.MaxPool1d],
          "actfn": [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
          "BN": [True,True,True,True]},

        decoder = {"type": ["deconv", "deconv","deconv","deconv"],
          "dim": [(256,128),(128,64),(64,32),(32,1)],
          "kernel": [(4,2,1),(6,2,2),(6,2,2),(8,2,3)],
          "pkernel": [(2, 1, 1),(2, 1, 1),(2, 1, 1),(2, 1, 1)],
          "pooling": [None,None,None,None],
          "actfn": [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
          "BN": [True,True,True,True]}
    )
# hidden1 = {"dim": [256], "Activate function": nn.ReLU(),"rest": nn.ReLU()},
#         hidden2 = {"dim": [256], "Activate function": nn.ReLU(0.2),"rest": nn.ReLU()},