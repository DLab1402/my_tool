import os
import json
import torch
import math
import numpy as np

class PPGDataset(torch.utils.data.Dataset):

    # lable_list = None --> one array
    # label_list = "AR" --> yes no
    # label_list = ["AF","PVC","PAC"] --> spercific

    def __init__(self, root,label_list = None):
        self.root = root
        self.labels = []
        self.signals = []
        file_list = os.listdir(root)
        self.sig = []
        self.lab = []
        self.count = 0
        for file in file_list:
            with open(self.root+"/"+file, 'r') as file:
                data= json.load(file)
                if label_list == None:
                    self.sig.append(data["Syn_PPG"])
                    self.lab.append(data["Syn_Label"])
                    self.count += 1
                elif label_list == "AR":
                    self.sig.append(data["Syn_PPG"])
                    self.lab.append([1 if x > 0 else 0 for x in data["Syn_Label"]])
                    self.count += 1
                elif isinstance(label_list,list):
                    self.sig.append(data["Syn_PPG"])
                    index = []
                    for i in label_list:
                        if i == "AF":
                            index.append(1)
                        elif i == "PVC":
                            index.append(2)
                        elif i == "PAC":
                            index.append(3)
                        elif i == "PAC-nhip-doi":
                            index.append(4)
                        elif i == "PAC-cap-doi":
                            index.append(5)
                        elif i == "PVC-nhip-doi":
                            index.append(6)
                        elif i == "PVC-cap-doi":
                            index.append(7)
                        elif i == "Block-AV-do-1":
                            index.append(8)
                        elif i == "Block-AV-do-2-mobitz-1":
                            index.append(9)
                        elif i == "Block-AV-do-2-mobitz-2":
                            index.append(10)
                        elif i == "Block-AV-do-3":
                            index.append(11)
                        elif i == "Noise":
                            index.append(12)
                        elif i == "Other":
                            index.append(13)
                        if i == "Normal":
                             index.append(14)
                    l = []
                    for i in index:
                        l.append([1 if x == i else 0 for x in data["Syn_Label"]])
                    self.lab.append(l)
                    self.count += 1
                    
                    
    def __getitem__(self, index):
        signal = torch.tensor([self.sig[index]],dtype=torch.float32)
        label = torch.tensor([self.lab[index]],dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.sig)
    
    if __name__ == "__main__":
        from torch.utils.data import Dataset,DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
        import matplotlib.pyplot as plt
        from ppg_labeled_load import PPGDataset
        dataset = PPGDataset('D:\ppg_project\Data\data_train',label_list = "AR")
        print(dataset.count)

        print(dataset[100][1].shape)
        for i in [400]:
            plt.plot(dataset[i][0].numpy().flatten())
            plt.plot(dataset[i][1].numpy().flatten())
            plt.show()