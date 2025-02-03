import os
import torch
import torchvision
from PIL import Image
# import numpy as np

transform = torchvision.transforms.ToTensor()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transform
        self.imgs = []
        
        for subdir, _, files in os.walk(root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    path = os.path.join(subdir, file)
                    self.imgs.append(path)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = img.resize((384,240))
        img = transform(img)
        return img, img
    
    def __len__(self):
        return len(self.imgs)
    
#Test script
if __name__ == "__main__":
    from sklearn.model_selection import KFold
    from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
    import sys
    root = '/workspace/data/results/raw/Ko đạt(366_367G)/images'
    dataset = ImageDataset(root)

    N = len(dataset)
    test_len = int(N*0.1)
    train_len = N-test_len
    
    train_data, test_data = random_split(dataset, [train_len, test_len])

    data = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    batch = next(iter(data))
    print(batch[0].shape)
    print(batch[0].max())
    print(batch[0].min())