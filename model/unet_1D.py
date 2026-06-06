"""
unet1d_general.py

A configurable 1D U-Net implementation.

Supported features:
- Per-level channel configuration
- Per-level kernel / stride / padding
- ReLU / LeakyReLU / GELU / ELU
- BatchNorm / InstanceNorm / None
- MaxPool / AvgPool / Strided Conv downsampling
- ConvTranspose / Upsample + Conv upsampling
- Automatic skip connections
- Shape visualization
- Parameter counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FACTORIES
# ============================================================

def build_activation(name):
    if name is None:
        return nn.Identity()

    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    elif name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)

    elif name == "gelu":
        return nn.GELU()

    elif name == "elu":
        return nn.ELU(inplace=True)

    elif name == "none":
        return nn.Identity()

    else:
        raise ValueError(f"Unsupported activation: {name}")


def build_norm(name, channels):

    if name is None:
        return nn.Identity()

    name = name.lower()

    if name == "batch":
        return nn.BatchNorm1d(channels)

    elif name == "instance":
        return nn.InstanceNorm1d(channels)

    elif name == "none":
        return nn.Identity()

    else:
        raise ValueError(f"Unsupported normalization: {name}")


# ============================================================
# CONV BLOCK
# ============================================================

class ConvBlock1D(nn.Module):

    def __init__(self,in_channels,out_channels,cfg):
        super().__init__()

        kernel = cfg.get("kernel", 3)
        stride = cfg.get("stride", 1)
        padding = cfg.get("padding", 1)
        norm = cfg.get("norm", "batch")
        activation = cfg.get("activation", "relu")

        self.block = nn.Sequential(

            nn.Conv1d(in_channels,out_channels,kernel_size=kernel,stride=stride,padding=padding),

            build_norm(norm, out_channels),
            build_activation(activation),

            nn.Conv1d(out_channels,out_channels,kernel_size=kernel,stride=1, padding=padding),

            build_norm(norm, out_channels),
            build_activation(activation)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# DOWNSAMPLE FACTORY
# ============================================================

def build_downsample(cfg, channels):

    if cfg is None:
        return None

    down_type = cfg["type"].lower()

    if down_type == "maxpool":

        return nn.MaxPool1d(
            kernel_size=cfg["kernel"],
            stride=cfg["stride"]
        )

    elif down_type == "avgpool":

        return nn.AvgPool1d(
            kernel_size=cfg["kernel"],
            stride=cfg["stride"]
        )

    elif down_type == "conv":

        return nn.Conv1d(channels,channels,kernel_size=cfg["kernel"],stride=cfg["stride"],padding=cfg["padding"])

    else:
        raise ValueError(
            f"Unsupported downsample type: {down_type}"
        )


# ============================================================
# UPSAMPLE FACTORY
# ============================================================

def build_upsample(in_channels,out_channels,cfg):

    if cfg is None:
        return None

    up_type = cfg["type"].lower()

    if up_type == "transpose":
        return nn.ConvTranspose1d(in_channels,out_channels,kernel_size=cfg["kernel"],stride=cfg["stride"])

    elif up_type == "upsample":
        mode = cfg.get("mode", "linear")
        return nn.Sequential(
            nn.Upsample(scale_factor=cfg["scale_factor"],mode=mode,align_corners=False),
            nn.Conv1d(in_channels,out_channels,kernel_size=1)
        )

    elif up_type == "nearest":
        return nn.Sequential(
            nn.Upsample(scale_factor=cfg["scale_factor"],mode="nearest"),
            nn.Conv1d(in_channels,out_channels,kernel_size=1)
        )

    else:
        raise ValueError(
            f"Unsupported upsample type: {up_type}"
        )


# ============================================================
# UNET
# ============================================================

class UNet1D(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.in_channels = config["Input channels"]
        self.out_channels = config["Output channels"]

        self.levels = config["Levels"]

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.vis = []

        # ====================================================
        # Encoder
        # ====================================================

        current_channels = self.in_channels

        for level in self.levels:

            out_channels = level["channels"]

            self.encoder.append(
                ConvBlock1D(
                    current_channels,
                    out_channels,
                    level["conv"]
                )
            )

            current_channels = out_channels

            if level["down"] is not None:

                self.downs.append(
                    build_downsample(
                        level["down"],
                        out_channels
                    )
                )

        # ====================================================
        # Decoder
        # ====================================================

        reversed_levels = self.levels[::-1]

        for i in range(len(reversed_levels) - 1):

            current_level = reversed_levels[i]
            next_level = reversed_levels[i + 1]

            self.ups.append(

                build_upsample(
                    current_level["channels"],
                    next_level["channels"],
                    next_level["up"]
                )
            )

            self.decoder.append(

                ConvBlock1D(
                    next_level["channels"] * 2,
                    next_level["channels"],
                    next_level["conv"]
                )
            )

        self.final_conv = nn.Conv1d(
            self.levels[0]["channels"],
            self.out_channels,
            kernel_size=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        self.vis.clear()
        skips = []
        down_idx = 0

        # -----------------------
        # Encoder
        # -----------------------

        for level_idx, block in enumerate(self.encoder):
            x = block(x)
            self.vis.append(x)

            if level_idx < len(self.levels) - 1:
                skips.append(x)
                x = self.downs[down_idx](x)
                self.vis.append(x)
                down_idx += 1

        skips = skips[::-1]

        # -----------------------
        # Decoder
        # -----------------------

        for up, dec, skip in zip(self.ups,self.decoder,skips):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x,size=skip.shape[-1],mode="linear",align_corners=False)

            x = torch.cat([skip, x],dim=1)
            x = dec(x)
            self.vis.append(x)

        x = self.final_conv(x)
        self.vis.append(x)
        return x

    # ========================================================
    # UTILITIES
    # ========================================================

    def visualize(self):

        print("\nFeature Maps")
        print("-" * 60)

        for idx, item in enumerate(self.vis):

            print(
                f"{idx:02d} : {list(item.shape)}"
            )

    def summary(self):

        print("\nUNet1D Summary")
        print("-" * 60)

        print(
            f"Input channels : {self.in_channels}"
        )

        print(
            f"Output channels: {self.out_channels}"
        )

        print(
            f"Depth          : {len(self.levels)}"
        )

        print("\nLevels")

        for idx, level in enumerate(self.levels):

            print(
                f"[{idx}] "
                f"Channels={level['channels']} "
                f"Kernel={level['conv']['kernel']}"
            )

    def count_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    config = {

        "Input channels": 1,
        "Output channels": 1,

        "Levels": [

            {
                "channels": 16,

                "conv": {
                    "kernel": 7,
                    "stride": 1,
                    "padding": 3,
                    "norm": "batch",
                    "activation": "relu"
                },

                "down": {
                    "type": "maxpool",
                    "kernel": 2,
                    "stride": 2
                },

                "up": {
                    "type": "transpose",
                    "kernel": 2,
                    "stride": 2
                }
            },

            {
                "channels": 32,

                "conv": {
                    "kernel": 5,
                    "stride": 1,
                    "padding": 2,
                    "norm": "batch",
                    "activation": "relu"
                },

                "down": {
                    "type": "avgpool",
                    "kernel": 2,
                    "stride": 2
                },

                "up": {
                    "type": "transpose",
                    "kernel": 2,
                    "stride": 2
                }
            },

            {
                "channels": 64,

                "conv": {
                    "kernel": 3,
                    "stride": 1,
                    "padding": 1,
                    "norm": "instance",
                    "activation": "gelu"
                },

                "down": {
                    "type": "maxpool",
                    "kernel": 2,
                    "stride": 2
                },

                "up": {
                    "type": "transpose",
                    "kernel": 2,
                    "stride": 2
                }
            },

            {
                "channels": 128,

                "conv": {
                    "kernel": 3,
                    "stride": 1,
                    "padding": 1,
                    "norm": "batch",
                    "activation": "relu"
                },

                "down": None,
                "up": None
            }
        ]
    }

    model = UNet1D(config)

    model.summary()

    x = torch.randn(
        8,
        1,
        512
    )

    y = model(x)

    print("\nInput Shape :", x.shape)
    print("Output Shape:", y.shape)

    model.visualize()

    print(
        "\nTrainable Parameters:",
        f"{model.count_parameters():,}"
    )