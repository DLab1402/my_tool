import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> SiLU Activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class C3Block(nn.Module):
    """CSP Bottleneck with 3 Convolutions (Inspired by CSPNet)."""
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.Sequential(
            *[ConvBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1) for _ in range(num_blocks)]
        )
        self.cv3 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.blocks(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) for multi-scale features."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.pool = nn.ModuleList([
            nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2) for _ in range(3)
        ])
        self.cv2 = ConvBlock(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pool(x) for pool in self.pool], dim=1))

class YOLOv8(nn.Module):
    """Simplified YOLOv8 Model."""
    def __init__(self, num_classes):
        super().__init__()
        # Backbone
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)  # Input Conv
        self.backbone1 = C3Block(32, 64, num_blocks=1)                    # Layer 1
        self.backbone2 = C3Block(64, 128, num_blocks=2)                   # Layer 2
        self.backbone3 = C3Block(128, 256, num_blocks=3)                  # Layer 3
        self.sppf = SPPF(256, 256)                                        # SPPF Layer

        # Neck
        self.neck = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )

        # Decoupled Head
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)        # Class scores
        self.obj_head = nn.Conv2d(128, 1, kernel_size=1)                  # Objectness score
        self.box_head = nn.Conv2d(128, 4, kernel_size=1)                  # Box coordinates [x, y, w, h]

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.backbone3(x)
        x = self.sppf(x)

        # Neck
        x = self.neck(x)

        # Decoupled Head
        cls = self.cls_head(x)
        obj = self.obj_head(x)
        box = self.box_head(x)

        # Final output: [x, y, w, h, obj_score, cls_probs]
        return torch.cat((box, obj, cls), dim=1)

if __name__ == "__main__":
    # Testing the model
    model = YOLOv8(num_classes=80)
    print(model)

    dummy_input = torch.randn(1, 3, 640, 640)  # Batch size 1, RGB image, 640x640
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [Batch, Predictions per feature map]
