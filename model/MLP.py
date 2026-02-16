import torch.nn as nn

class MLP(nn.Module):
    vis = []
    para = {
            "dim":               [],
            "Activate function": [],
            "Activate last":     []
        }
    def __init__(self,para):
        super(MLP, self).__init__()
        self.para = para
        self.layers = nn.ModuleList()
        self.structure_calculate()

    def structure_calculate(self):
        for dim in self.para["dim"]:
            self.layers.append(nn.Linear(dim[0],dim[1]))
            if dim != self.para["dim"][-1]:
                if self.para["Activate function"] == "ReLU":
                    self.layers.append(nn.ReLU())
                elif self.para["Activate function"] == "LeakyReLU":
                    self.layers.append(nn.LeakyReLU(0.2))
                elif self.para["Activate function"] == "Tanh":
                    self.layers.append(nn.Tanh())
                elif self.para["Activate fuction"] == "Sigmoid":
                    self.layers.append(nn.Sigmoid())
        if self.para["Activate last"] == "ReLU":
            self.layers.append(nn.ReLU())
        elif self.para["Activate last"] == "LeakyReLU":
            self.layers.append(nn.LeakyReLU(0.2))
        elif self.para["Activate last"] == "Tanh":
            self.layers.append(nn.Tanh())
        elif self.para["Activate last"] == "Sigmoid":
            self.layers.append(nn.Sigmoid())

    def forward(self, noise):
        self.vis.clear()
        out = noise
        for layer in self.layers:
            out = layer(out)
            self.vis.append(out)
        return out