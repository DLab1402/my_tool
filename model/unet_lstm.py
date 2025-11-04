import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_base_1D import unet_base_1D

class unet_lstm(nn.Module):
    
    vis = []
    para = {"Serie length":         [],
            "Input size":           [],
            "Hidden size":          [],
            "Number layers":        [],
            "Encoder structure":    [], 
            "Decoder structure":    [],
            "Kernel size":          [],
            "Kernel encoder":       [],
            "Kernel decoder":       [],
            "Stride":               [],
            "Activate function":    []
    }

    def __init__(self,para):
        super(unet_lstm, self).__init__()
        unet_para = {
            "Serie length":         para["Serie length"],
            "Encoder structure":    para["Encoder structure"],
            "Decoder structure":    para["Decoder structure"],
            "Kernel size":          para["Kernel size"],
            "Stride":               para["Stride"],
            "Activate function":    para["Activate function"]
        }
        self.unet = unet_base_1D(unet_para)
        self.lstm =  nn.LSTM(para["Input size"], para["Hidden size"], para["Number layers"], batch_first=True)
    
    def structure_calculate(self,visualize=False):
        
    
    def lstm_block(self):
       
        
    def forward(self, x):
       
    def visualizer(self):
        for item in self.vis:
            print(item.shape)
# Example usage
if __name__ =="__main__":
    para = {"Serie length":         800,
            "Input size":           1,
            "Hidden size":          512, # changed hidden size to match with the struct
            "Number layers":        2,
            "Encoder structure":    [1,64,128,256,512], 
            "Decoder structure":    [512,256,128,64,1],
            "Kernel size":          [3,3,3,3,3,3,3,3,3],
            "Kernel encoder":       [8,8,8,8],
            "Kernel decoder":       [4,4,4,4],
            "Stride":               [2,2,2,2],
            "Activate function":    nn.Tanh()
    }
    input_tensor = torch.rand((4, 1, 1024))
    model = unet_lstm(para)
    model.structure_calculate(True)
    output = model(input_tensor)
    model.visualizer()