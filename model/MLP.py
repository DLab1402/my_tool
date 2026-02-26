import torch.nn as nn

class MLP(nn.Module):
    vis = []
    para = {
            "dim":               [],
            "Activate function": [],
            "Activate last":     None,
            "BatchNorm":   False
        }
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.layers = nn.ModuleList()
        self.build()

    def build(self):
        dims = self.para["dim"]
        act_fn = self.para.get("Activate function", None)
        act_last = self.para.get("Activate last", None)
        use_bn = self.para.get("BatchNorm", False)

        for i, dim in enumerate(dims):
            in_dim, out_dim = dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i != len(dims) - 1 and act_fn is not None:
                self.layers.append(type(act_fn)())
            if use_bn:
                self.layers.append(nn.BatchNorm1d(out_dim))

        if act_last is not None:
            self.layers.append(act_last)
            
    def forward(self, x):
        self.vis.clear()
        for layer in self.layers:
            x = layer(x)
            self.vis.append(x)
        return x

if __name__ == "__main__":
    para = {
        "dim": [(10, 20), (20, 30)],
        "Activate function": nn.ReLU(),
        "BatchNorm": True
    }
    model = MLP(para)
    print(model)