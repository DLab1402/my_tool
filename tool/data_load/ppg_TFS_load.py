import os
import torch
import torchvision
import numpy as np
from PIL import Image
transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=transform):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.co_0 = 0
        self.co_1 = 0
        self.co_2 = 0
        
        for subdir, _, files in os.walk(root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    path = os.path.join(subdir, file)
                    label = os.path.basename(file)
                    if int(label[-5]) == 0:
                      self.imgs.append((path,np.array([1.,0.])))
                      self.co_0 = self.co_0+1
                    if int(label[-5]) == 1:
                      self.imgs.append((path,np.array([0.,1.])))
                      self.co_1 = self.co_1+1
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
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