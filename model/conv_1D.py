import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.vis = []
        self.build(para)

    def build(self, para):
        types = para["type"]
        dim = para["dim"]
        ker = para["kernel"]
        pker = para["pkernel"]
        pool = para["pooling"]
        act = para["actfn"]
        BN = para["BN"]

        for i in range(len(types)):
            layers = []

            if types[i] == "conv":
                layers.append(nn.Conv1d(dim[i][0],dim[i][1],ker[i][0],ker[i][1],ker[i][2]))

            elif types[i] == "deconv":
                layers.append(nn.ConvTranspose1d(dim[i][0],dim[i][1],ker[i][0],ker[i][1],ker[i][2],))

            if BN[i]:
                layers.append(nn.BatchNorm1d(dim[i][1]))

            layers.append(act[i])
            # ----- Pool -----
            layers.append(pool[i](kernel_size=pker[i][0],stride=pker[i][1],padding=pker[i][2]))

            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        self.vis.clear()
        for block in self.blocks:
            x = block(x)
            self.vis.append(x)
        return x

if __name__ == "__main__":

    # -------- Network parameters --------
    para = {
        "type": ["conv", "conv", "conv"],
        "dim": [(1, 16),(16, 32),(32, 64)],
        "kernel": [(3, 1, 1),(3, 1, 1),(3, 1, 1)],
        "pkernel": [(2, 2, 0),(2, 2, 0),(2, 2, 0)],
        "pooling": [nn.MaxPool1d,nn.MaxPool1d,nn.MaxPool1d],
        "actfn": [nn.ReLU(),nn.LeakyReLU(0.1),nn.ReLU()],
        "BN": [True, False, True],
    }

    # -------- Create model --------
    model = CNN1D(para)

    print("\nMODEL STRUCTURE\n")
    print(model)


    # -------- Dummy input --------
    # (batch, channel, length)
    x = torch.randn(8, 1, 100)

    print("\nInput shape:", x.shape)


    # -------- Forward --------
    y = model(x)

    print("\nFinal output shape:", y.shape)


    # -------- Intermediate outputs --------
    print("\nLayer-wise outputs:")
    for i, v in enumerate(model.vis):
        print(f"Layer {i}: {v.shape}")


    # -------- Backprop test --------
    print("\nGradient check...")

    loss = y.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: OK")