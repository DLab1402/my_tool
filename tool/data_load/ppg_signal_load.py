import json
import torch
import math
import numpy as np

class PPGDataset(torch.utils.data.Dataset):

    def __init__(self, root,preprocessing = None):
        self.root = root
        self.labels = []
        self.signals = []
        with open(root, 'r') as file:
            data = json.load(file)
            print("The total data points: {}".format(len(data)))
        for item in data:
            if preprocessing == None:
                self.signals.append(item[0])
                self.labels.append(item[1])
            else:
                tem = item
                tem = preprocessing(tem)
                self.signals.append(tem[0])
                self.labels.append(tem[1])

    def __getitem__(self, index):
        signal = torch.tensor([self.signals[index]],dtype=torch.float32)
        label = torch.tensor([self.labels[index]],dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.labels)