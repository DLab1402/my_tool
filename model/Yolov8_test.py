import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class C3Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.cv2 = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.blocks = nn.Sequential(*[
            ConvBlock(out_channels // 2, out_channels // 2, 3, 1, 1) for _ in range(num_blocks)
        ])
        self.cv3 = ConvBlock(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.blocks(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.pool = nn.ModuleList([
            nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2) for _ in range(3)
        ])
        self.cv2 = ConvBlock(out_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pool(x) for pool in self.pool], dim=1))

class YOLOv8(nn.Module):

    para = {
        "Input channels": 3,
        "Stem channels": 32,
        "Backbone blocks": [(32, 64, 1), (64, 128, 2), (128, 256, 3)],
        "SPPF channels": (256, 256),
        "Neck blocks": [(256, 128), (128, 128)],
        "Head channels": {"cls": 80, "obj": 1, "box": 4}
    }

    def __init__(self, config=None):
        super().__init__()
        if config:
            for key in self.para:
                self.para[key] = config[key]

        # Stem
        self.stem = ConvBlock(
            self.para["Input channels"],
            self.para["Stem channels"],
            kernel_size=3, stride=1, padding=1
        )

        # Backbone
        self.backbone = nn.Sequential()
        for i, (in_c, out_c, num_blocks) in enumerate(self.para["Backbone blocks"]):
            self.backbone.add_module(f"backbone{i+1}", C3Block(in_c, out_c, num_blocks))

        # SPPF
        in_sppf, out_sppf = self.para["SPPF channels"]
        self.sppf = SPPF(in_sppf, out_sppf)

        # Neck
        self.neck = nn.Sequential()
        for i, (in_c, out_c) in enumerate(self.para["Neck blocks"]):
            self.neck.add_module(f"neck{i+1}", ConvBlock(in_c, out_c, 3, 1, 1))

        # Head
        head_in = self.para["Neck blocks"][-1][1]
        self.cls_head = nn.Conv2d(head_in, self.para["Head channels"]["cls"], 1, 1, 0)
        self.obj_head = nn.Conv2d(head_in, self.para["Head channels"]["obj"], 1, 1, 0)
        self.box_head = nn.Conv2d(head_in, self.para["Head channels"]["box"], 1, 1, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.sppf(x)
        x = self.neck(x)

        cls = self.cls_head(x)
        obj = self.obj_head(x)
        box = self.box_head(x)

        return torch.cat((box, obj, cls), dim=1)

    def print_summary(self):
        print("\n YOLOv8 Summary:")
        print(f"Input channels: {self.para['Input channels']}")
        print(f"Stem: {self.para['Stem channels']} channels")
        print(f"Backbone:")
        for i, (in_c, out_c, n) in enumerate(self.para["Backbone blocks"]):
            print(f"Block {i+1}: {in_c} --> {out_c}, {n} convs")
        print(f"SPPF: {self.para['SPPF channels'][0]} --> {self.para['SPPF channels'][1]}")
        print(f"Neck:")
        for i, (in_c, out_c) in enumerate(self.para["Neck blocks"]):
            print(f"Neck {i+1}: {in_c} --> {out_c}")
        print(f"Head: cls={self.para['Head channels']['cls']}, obj={self.para['Head channels']['obj']}, box={self.para['Head channels']['box']}")

# ---------------------- Run Example ----------------------

if __name__ == "__main__":
    model = YOLOv8()
    model.print_summary()
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print("Output shape:", output.shape)
