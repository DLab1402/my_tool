import os
import torch
import torchvision
# import numpy as np
import cv2

transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=transform, resize = None):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.co_0 = 0
        self.co_1 = 0
        self.co_2 = 0
        self.resize = resize
        
        for subdir, _, files in os.walk(root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    path = os.path.join(subdir, file)
                    self.imgs.append(path)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = cv2.imread(path)
        if self.resize is  not None:
            img = cv2.resize(img, self.resize)
        if self.transform is not None:
            img = self.transform(img)
        return img, img
    
    def __len__(self):
        return len(self.imgs)

#Test script
# from sklearn.model_selection import KFold
# from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
# root = 'Train_data'
# dataset = ImageDataset(root, transform=transform)
# train_data, test_data = random_split(dataset, [2446, 612])
# print(dataset.co_0)
# print(dataset.co_1)
# print(dataset.co_2)
# data = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
# for inputs, labels in data:
#     print(inputs.shape)
# print(sys.getsizeof(train_data[0]))