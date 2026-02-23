import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    vis = []
    para = [{
        "type": "conv",
        "channels": [64,64],
        "stride": 2,
        "padding": 0,
        "kernel": 3,
        "activate function": nn.ReLU(),
        "pool name":F.max_pool1d,
        "pool kernel": 2,
        "pool stride": 2,
        "pool padding": 0,
        "BatchNorm": False,
    }]
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.layers = nn.ModuleList()
        self.structure_calculate()

    def structure_calculate(self):
        for ch in self.para:
            if ch["type"] == "conv":
                layer  = nn.Sequential(
                    nn.Conv1d(ch["channels"][0], ch["channels"][1], kernel_size=ch["kernel"], stride=ch["stride"], padding=ch["padding"]),
                    ch["activate function"]
                )
            elif ch["type"] == "deconv":
                layer  = nn.Sequential(
                    nn.ConvTranspose1d(ch["channels"][0], ch["channels"][1], kernel_size=ch["kernel"], stride=ch["stride"], padding=ch["padding"]),
                    ch["activate function"]
                )
            if ch["BatchNorm"]:
                layer.add_module("BatchNorm", nn.BatchNorm1d(ch["channels"][1]))
            self.layers.append(layer)

    def forward(self, x):
        self.vis.clear()
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if self.para[i]["pool name"] is not None:
                x = self.para[i]["pool name"](x, kernel_size=self.para[i]["pool kernel"], stride=self.para[i]["pool stride"], padding=self.para[i]["pool padding"])
            self.vis.append(x)

        return x


if __name__ == "__main__":
    para = [{
        "channels": [1, 64],
        "stride": 1,
        "padding": 1,
        "kernel": 3,
        "activate function": nn.ReLU(),
        "pool name": F.max_pool1d,
        "pool kernel": 2,
        "pool stride": 2,
        "pool padding": 0,
        "BatchNorm": True
    }]
    model = Conv1DNet(para)
    input_data = torch.randn(16, 1, 128)  # Batch size of 16, 1 channel, sequence length of 128
    output = model(input_data)
    print(output.shape)