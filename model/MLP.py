import torch.nn as nn

class MLP(nn.Module):
    vis = []
    para = {
            "dim":               [],
            "Activate function": [],
            "Activate last":     None
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
                self.layers.append(self.para["Activate function"])
        if self.para["Activate last"] != None:
            self.layers.append(self.para["Activate last"])
            
    def forward(self, noise):
        self.vis.clear()
        out = noise
        for layer in self.layers:
            out = layer(out)
            self.vis.append(out)
        return out