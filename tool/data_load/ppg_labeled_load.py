import os
import json
import torch
import math
import numpy as np

class PPGDataset(torch.utils.data.Dataset):

    def __init__(self, root,preprocessing = None):
        self.root = root
        self.labels = []
        self.signals = []
        self.processor = preprocessing
        file_list = os.listdir(root)
        self.sig = []
        self.lab = []
        count = 0
        for file in file_list:
            with open(self.root+"/"+file, 'r') as file:
                data= json.load(file)
                if self.processor == None:
                    if np.sum(data["Syn_Label"])>0:
                        self.sig.append(data["Syn_PPG"])
                        # self.lab.append([1 if x > 0 else x for x in data["Syn_Label"]])
                        self.lab.append(data["Syn_Label"])
                    else:
                        if count < 500:
                            count = count+1
                            self.sig.append(data["Syn_PPG"])
                            self.lab.append([1 if x > 0 else x for x in data["Syn_Label"]])
                else:
                    s,l = self.processor(data)
                    self.sig.append(s)
                    self.lab.append(l)

    def __getitem__(self, index):
        signal = torch.tensor([self.sig[index]],dtype=torch.float32)
        label = torch.tensor([self.lab[index]],dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.sig)