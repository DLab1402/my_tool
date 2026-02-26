import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, layer_type,in_ch, out_ch,
                 kernel=3, stride=1, padding=0,
                 output_padding=0,
                 activation=nn.ReLU,
                 use_bn=False,
                 pool_kernel=None, pool_stride=None,
                 upsample=False):

        super().__init__()

        layers = []

        if layer_type == "conv":
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel, stride, padding)
            )

        elif layer_type == "deconv":
            layers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            )

        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))

        if activation is not None:
            layers.append(activation())

        if pool_kernel is not None:
            layers.append(pool_kernel(pool_kernel, pool_stride))

        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv1DNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.vis = []

        for cfg in config:
            block = ConvBlock1D(
                layer_type=cfg["type"],
                in_ch=cfg["in_channels"],
                out_ch=cfg["out_channels"],
                kernel=cfg.get("kernel", 3),
                stride=cfg.get("stride", 1),
                padding=cfg.get("padding", 0),
                output_padding=cfg.get("output_padding", 0),
                activation=cfg.get("activation", nn.ReLU),
                use_bn=cfg.get("batchnorm", False),
                pool_kernel=cfg.get("pool_kernel", nn.MaxPool1d),
                pool_stride=cfg.get("pool_stride", None),
                upsample=cfg.get("upsample", False)
            )

            self.layers.append(block)

    def forward(self, x):
        self.vis.clear()
        for layer in self.layers:
            x = layer(x)
            self.vis.append(x)
        return x


if __name__ == "__main__":

    config = [
        # Encoder
        {
            "type": "conv",
            "in_channels": 1,
            "out_channels": 64,
            "kernel": 3,
            "stride": 2,
            "padding": 1,
            "batchnorm": True
        },

        # Decoder (transpose conv)
        {
            "type": "deconv",
            "in_channels": 64,
            "out_channels": 1,
            "kernel": 3,
            "stride": 2,
            "padding": 1,
            "output_padding": 1,  # important!
            "activation": None
        }
    ]

    model = Conv1DNet(config)

    x = torch.randn(16, 1, 128)
    out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)