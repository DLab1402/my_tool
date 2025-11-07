import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_base_1D import unet_base_1D

class unet_lstm(nn.Module):
    
    vis = []
    para = {"Serie length":         [], #The length of serie
            "Encoder structure":    [], #N+1
            "Decoder structure":    [], #N+1
            "Kernel size":          [], #2N+1, Kernel size for conv layers of both encoder and decoder
            "Down kernel":          [], #N, Kernel of downsampling pooling conv
            "Up kernel":            [], #N, Kernel of upsampling pooling conv
            "Stride":               [], #N, Stride for both downsampling and upsampling
            "Activate function":    [],
            "Input size":           [],
            "Hidden size":          [],
            "Number layers":        []
    }

    def __init__(self,para):
        super(unet_lstm, self).__init__()
        unet_para = {
            "Serie length":         para["Serie length"],
            "Encoder structure":    para["Encoder structure"],
            "Decoder structure":    para["Decoder structure"],
            "Kernel size":          para["Kernel size"],
            "Down kernel":          para["Down kernel"],
            "Up kernel":            para["Up kernel"],
            "Stride":               para["Stride"],
            "Activate function":    para["Activate function"]
        }
        self.unet = unet_base_1D(unet_para)
        self.lstm = nn.LSTM(para["Input size"], para["Hidden size"], para["Number layers"], batch_first=True, proj_size=1)
    
    def structure_calculate(self,visualize=False):
        if visualize:
            self.unet.structure_calculate(True)
        
    def forward(self, x):
        self.vis = []
        out = self.unet(x)
        print(out.shape)
        self.vis.append(self.unet.vis)
        
        out, (hn, cn) = self.lstm(out.permute(0,2,1))
        self.vis.append(out)
        print(out.shape)
        return out.permute(0,2,1)

    def visualizer(self):
        pass
# Example usage
if __name__ =="__main__":
    para = {"Serie length":         800,
            "Input size":           50,
            "Hidden size":          20, # changed hidden size to match with the struct
            "Number layers":        5,
            "Encoder structure":    [1,64,128,256,512], 
            "Decoder structure":    [512,256,128,64,50],
            "Kernel size":          [3,3,3,3,3,3,3,3,3],
            "Down kernel":          [8,8,8,8],
            "Up kernel":            [4,4,4,4],
            "Stride":               [2,2,2,2],
            "Activate function":    nn.Tanh()
    }
    input_tensor = torch.rand((4, 1, 1024))
    model = unet_lstm(para)
    model.structure_calculate(True)
    output = model(input_tensor)
    model.visualizer()