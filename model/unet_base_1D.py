import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_base_1D(nn.Module):

    vis = []
    para = {
        "Serie length":         [],
        "Input size":           [],
        "Encoder structure":    [],
        "Decoder structure":    [],
        "Kernel size":          [],
        "Stride":               [],
        "Activate function":    []
    }

    def __init__(self, config):
        super(unet_base_1D, self).__init__()

        # Gán cấu hình từ config vào self.para
        for key in self.para:
            self.para[key] = config[key]

        self.encoder_channels = self.para["Encoder structure"]
        self.decoder_channels = self.para["Decoder structure"]
        self.kernel_sizes = self.para["Kernel size"]
        self.strides = self.para["Stride"]
        self.activation = self.para["Activate function"]
        self.input_size = self.para["Input size"]
        self.serie_length = self.para["Serie length"]
        self.depth = len(self.encoder_channels) - 1

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        def conv_block(in_c, out_c, kernel_size):
            padding = (kernel_size - 1) // 2
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_c),
                self.activation,
                nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_c),
                self.activation
            )

        for i in range(self.depth):
            self.encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1], self.kernel_sizes[i]))

        self.bottleneck = conv_block(self.encoder_channels[-1], self.decoder_channels[0], self.kernel_sizes[self.depth])

        for i in range(self.depth):
            up_in = self.decoder_channels[i]
            up_out = self.encoder_channels[self.depth - 1 - i]
            dec_in = up_out + up_in
            dec_out = self.decoder_channels[i + 1]

            self.upconv_layers.append(nn.ConvTranspose1d(up_in, up_out, kernel_size=self.strides[i], stride=self.strides[i]))
            self.decoder_layers.append(conv_block(dec_in, dec_out, self.kernel_sizes[self.depth + 1 + i]))

        self.final_conv = nn.Conv1d(self.decoder_channels[-1], self.input_size, kernel_size=1)

    def forward(self, x):
        self.vis.clear()
        skips = []
        out = x
        self.vis.append(out)

        for i in range(self.depth):
            out = self.encoder_layers[i](out)
            skips.append(out)
            out = F.max_pool1d(out, kernel_size=self.strides[i], stride=self.strides[i])
            self.vis.append(out)

        out = self.bottleneck(out)
        self.vis.append(out)

        for i in range(self.depth):
            out = self.upconv_layers[i](out)
            skip = skips[self.depth - 1 - i]
            if out.size(2) != skip.size(2):
                out = F.interpolate(out, size=skip.size(2), mode='linear', align_corners=False)
            out = torch.cat((skip, out), dim=1)
            out = self.decoder_layers[i](out)
            self.vis.append(out)

        out = self.final_conv(out)
        self.vis.append(out)
        return out

    def print_structure(self):
        Lout = [self.serie_length]
        for s in self.strides:
            Lout.append(int(Lout[-1] / s))
        print(Lout)

        encoder_pairs = [[self.encoder_channels[i], self.encoder_channels[i+1]] for i in range(self.depth)]
        encoder_kernels = self.kernel_sizes[:self.depth]
        encoder_paddings = [float((k - 1) // 2) for k in encoder_kernels]
        print([encoder_pairs, encoder_kernels, encoder_paddings])

        bottleneck_pair = [self.encoder_channels[-1], self.decoder_channels[0]]
        bottleneck_kernel = self.kernel_sizes[self.depth]
        bottleneck_padding = float((bottleneck_kernel - 1) // 2)
        print([bottleneck_pair, bottleneck_kernel, bottleneck_padding])

        decoder_pairs = []
        for i in range(self.depth):
            in_c = self.encoder_channels[self.depth - 1 - i] + self.decoder_channels[i]
            out_c = self.decoder_channels[i + 1]
            decoder_pairs.append([in_c, out_c])
        decoder_kernels = self.kernel_sizes[self.depth + 1:]
        decoder_paddings = [float((k - 1) // 2) for k in decoder_kernels]
        print([decoder_pairs, decoder_kernels, decoder_paddings])

        down_kernel = [8] * self.depth
        down_stride = self.strides
        down_padding = [float((k - 1) // 2) for k in down_kernel]
        print([down_kernel, down_stride, down_padding])

        up_kernel = [4] * self.depth
        up_stride = self.strides
        up_padding = [float((k - 1) // 2) for k in up_kernel]
        print([up_kernel, up_stride, up_padding])

        for feature in self.vis:
            print(feature.shape)

# ---------------------- Run Example ----------------------

if __name__ == "__main__":
    config = {
        "Serie length":         800,
        "Input size":           1,
        "Encoder structure":    [1, 64, 128, 256, 512],
        "Decoder structure":    [512, 256, 128, 64, 100],
        "Kernel size":          [3] * 9,
        "Stride":               [2, 2, 2, 2],
        "Activate function":    nn.GELU()
    }

    input_tensor = torch.rand((4, config["Input size"], config["Serie length"]))
    model = unet_base_1D(config)
    output = model(input_tensor)
    model.print_structure()
    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

"""

if __name__ == "__main__":
    config = {
    "Serie length":         800,  # độ dài chuỗi đầu vào
    "Input size":           100,  # số kênh đầu vào (ví dụ: 100 đặc trưng)
    
    # Encoder: 5 tầng, tăng dần số kênh
    "Encoder structure":    [100, 64, 128, 256, 512, 1024],
    
    # Decoder: 5 tầng, giảm dần số kênh, kết thúc bằng số kênh mong muốn đầu ra
    "Decoder structure":    [1024, 512, 256, 128, 64, 100],
    
    # Tổng số kernel size = encoder + bottleneck + decoder = 11
    "Kernel size":          [3] * 11,
    
    # Stride cho mỗi tầng encoder (4 bước downsampling)
    "Stride":               [2, 2, 2, 2, 2],
    
    # Hàm kích hoạt hiện đại, hiệu quả cao
    "Activate function":    nn.GELU()
    }
    input_tensor = torch.rand((4, config["Input size"], config["Serie length"]))
    model = unet_base_1D(config)
    output = model(input_tensor)
    model.print_structure()
    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

"""