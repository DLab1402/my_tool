import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    #feature map
    self.fw_cnv1 = []
    self.fw_cnv2 = []
    self.fw_cnv3 = []
    self.fw_cnv4 = []
    self.fw_cnv5 = []
    self.fw_cnv6 = []
    self.fw_cnv7 = []
    #self.act1 = torch.nn.LeakyReLU(0.1)
    #...
    self.bn1 = nn.BatchNorm2d(3)
    self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool1 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 1)
    #...
    self.bn2 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 8, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool2 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 2)
    #...
    self.bn3 = nn.BatchNorm2d(8)
    self.conv3 = nn.Conv2d(8, 8, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool3 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 1)
    #...
    self.bn4 = nn.BatchNorm2d(8)
    self.conv4 = nn.Conv2d(8, 16, kernel_size= (5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool4 = nn.MaxPool2d(kernel_size = (5,9), padding = (2,4), stride = 2)
    #...
    self.bn5 = nn.BatchNorm2d(16)
    self.conv5 = nn.Conv2d(16, 16, kernel_size =(3,5), padding = (1,2), stride = 1,  bias=False)
    self.pool5 = nn.MaxPool2d(kernel_size = (3,5), padding = (1,2), stride = 2)
    #...
    self.bn6 = nn.BatchNorm2d(16)
    self.conv6 = nn.Conv2d(16, 16, kernel_size =(1,3), padding = (0,1), stride = 1,  bias=False)
    self.pool6 = nn.MaxPool2d(kernel_size = (1,3), padding = (0,1), stride = 2)
    #...
    self.bn7 = nn.BatchNorm2d(16)
    self.conv7 = nn.Conv2d(16, 16, kernel_size =(1,3), padding = (0,1), stride = 1,  bias=False)
    self.pool7 = nn.MaxPool2d(kernel_size = (1,3), padding = (0,1), stride = 2)
    #...
    self.linear1 = nn.Linear(1024,256)
    # self.linear2 = nn.Linear(640,640)
    self.linear3 = nn.Linear(256,150)
    # self.linear4 = nn.Linear(150,75)
    self.linear5 = nn.Linear(150,2)
    self.act = nn.LeakyReLU(0.2)
    self.sm = nn.Softmax(dim = -1)
  def forward(self, input):
    x = self.act(self.conv1(input))
    x = self.pool1(x)
    self.fw_cnv1 = x
    # print(x.shape)
    x = self.act(self.conv2(x))
    x = self.pool2(x)
    self.fw_cnv2 = x
    # print(x.shape)
    x = self.act(self.conv3(x))
    x = self.pool3(x)
    self.fw_cnv3 = x
    # print(x.shape)
    x = self.act(self.conv4(x))
    x = self.pool4(x)
    self.fw_cnv4 = x
    # print(x.shape)
    x = self.act(self.conv5(x))
    x = self.pool5(x)
    self.fw_cnv5 = x
    # print(x.shape)
    x = self.act(self.conv6(x))
    x = self.pool6(x)
    self.fw_cnv6 = x
    # print(x.shape)
    x = self.act(self.conv7(x))
    x = self.pool7(x)
    self.fw_cnv7 = x
    # print(x.shape)
    x = torch.flatten(x, start_dim=1, end_dim=3)
    x = self.act(self.linear1(x))
    # x = self.act(self.linear2(x))
    x = self.act(self.linear3(x))
    # x = self.act(self.linear4(x))
    x = self.sm(self.linear5(x))
    return x
  
class CNN_LSTM(nn.Module):
  def __init__(self):
    super(CNN_LSTM, self).__init__()
    self.fm_conv1 = []
    self.fw_conv2 = []
    self.fw_conv3 = []
    self.sh = []
    self.bn = nn.BatchNorm2d(0)
    self.act = nn.LeakyReLU(0.2)
    # CNN layer
    self.bn1 = nn.BatchNorm2d(3)
    self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool1 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 2)
    
    self.bn2 = nn.BatchNorm2d(4)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool2 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 1)

    self.bn3 = nn.BatchNorm2d(8)
    self.conv3 = nn.Conv2d(16, 16, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool3 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 2)

    # LSTM layer
    self.lstm = nn.LSTM(input_size= 512, hidden_size= 100, num_layers=2, batch_first=True)
    self.fc1 = nn.Linear(1500,750)
    self.fc2 = nn.Linear(750,300)
    self.fc3 = nn.Linear(300,100)
    self.fc4 = nn.Linear(100,50)
    self.fc5 = nn.Linear(100,2)
    self.sm = nn.Softmax(dim = -1)
  def forward(self, input):
    x = self.act(self.conv1(input))
    x = self.pool1(x)
    self.fw_conv1 = x
    # print(x.shape)
    x = self.act(self.conv2(x))
    x = self.pool2(x)
    self.fw_conv2 = x
    # print(x.shape)
    x = self.act(self.conv3(x))
    x = self.pool3(x)
    self.fw_conv3 = x
    # print(x.shape)
    x = torch.flatten(x, start_dim=1, end_dim=2)
    # print(x.shape)
    x = x.transpose(1,2)
    out,(h,c) = self.lstm(x)
    # print(out.shape)
    x = out[:,-1]
    self.sh = x
    # print(x.shape)
    x = self.sm(self.fc5(x))
    return x