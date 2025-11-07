import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_base_1D(nn.Module):
    #visual feature map
    vis = []

    #length of series
    Lout = []

    #conv para
    strides = []
    
    #down para
    down_stride = []
    down_kernel = []
    down_padding = []
    
    #up para
    up_stride = []
    up_kernel = []
    up_padding = []

    para = {
        "Serie length":         [], #The length of serie
        "Encoder structure":    [], #N+1
        "Decoder structure":    [], #N+1
        "Kernel size":          [], #2N+1, Kernel size for conv layers of both encoder and decoder
        "Down kernel":          [], #N, Kernel of downsampling pooling conv
        "Up kernel":            [], #N, Kernel of upsampling pooling conv
        "Stride":               [], #N, Stride for both downsampling and upsampling
        "Activate function":    []
    }

    def __init__(self, config):
        super(unet_base_1D, self).__init__()

        for key in self.para:
            self.para[key] = config[key]

        self.encoder_channels = self.para["Encoder structure"]
        self.decoder_channels = self.para["Decoder structure"]
        self.kernel_sizes = self.para["Kernel size"]
        self.down_kernel = self.para["Down kernel"]
        self.up_kernel = self.para["Up kernel"]   
        self.strides = self.para["Stride"]
        self.activation = self.para["Activate function"]
        self.serie_length = self.para["Serie length"]
        self.depth = len(self.encoder_channels) - 1

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.structure_calculate()

        for i in range(self.depth):
            self.encoder_layers.append(self.conv_block(self.encoder_channels[i], self.encoder_channels[i+1], self.kernel_sizes[i]))

        self.bottleneck = self.conv_block(self.encoder_channels[-1], self.decoder_channels[0], self.kernel_sizes[self.depth])

        for i in range(self.depth):
            up_in = self.decoder_channels[i]
            up_out = self.encoder_channels[self.depth - 1 - i]
            dec_in = up_out + up_in
            dec_out = self.decoder_channels[i + 1]

            self.upconv_layers.append(nn.ConvTranspose1d(up_in, up_out, kernel_size=self.up_kernel[i], stride=self.strides[len(self.strides)-i-1],padding=self.up_padding[i]))
            self.decoder_layers.append(self.conv_block(dec_in, dec_out, self.kernel_sizes[self.depth + 1 + i]))
    
    def conv_block(self,in_c, out_c, kernel_size):
        #Hold the same length after conv
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
            self.activation,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
            self.activation
        )
    
    def structure_calculate(self,visualize=False):
        Lout = [self.serie_length]
        for s in self.strides:
            Lout.append(int(Lout[-1] / s))
        
        encoder_pairs = [[self.encoder_channels[i], self.encoder_channels[i+1]] for i in range(self.depth)]
        encoder_kernels = self.kernel_sizes[:self.depth]
        encoder_paddings = [float((k - 1) // 2) for k in encoder_kernels]
        
        bottleneck_pair = [self.encoder_channels[-1], self.decoder_channels[0]]
        bottleneck_kernel = self.kernel_sizes[self.depth]
        bottleneck_padding = float((bottleneck_kernel - 1) // 2)
        
        decoder_pairs = []
        for i in range(self.depth):
            in_c = self.encoder_channels[self.depth - 1 - i] + self.decoder_channels[i]
            out_c = self.decoder_channels[i + 1]
            decoder_pairs.append([in_c, out_c])
        decoder_kernels = self.kernel_sizes[self.depth + 1:]
        decoder_paddings = [float((k - 1) // 2) for k in decoder_kernels]

        padding_conv_cal = lambda Lin,Lout,kernel,stride: int(0.5*((Lout-1)*stride-Lin+kernel))
        padding_convT_cal = lambda Lin,Lout,kernel,stride: int(0.5*((Lin-1)*stride-Lout+kernel))

        down_stride = self.strides
        self.down_padding = [padding_conv_cal(Lout[k],Lout[k+1],self.down_kernel[k],down_stride[k]) for k in range(len(self.down_kernel))]

        up_stride = list(reversed(self.strides))
        self.up_padding = [padding_convT_cal(Lout[len(Lout)-k-1],Lout[len(Lout)-k-2],self.up_kernel[k],up_stride[k]) for k in range(len(self.up_kernel))]

        if visualize == True:
            print(Lout)
            print([encoder_pairs, encoder_kernels, encoder_paddings])
            print([bottleneck_pair, bottleneck_kernel, bottleneck_padding])
            print([decoder_pairs, decoder_kernels, decoder_paddings])
            print([self.down_kernel, down_stride, self.down_padding])
            print([self.up_kernel, up_stride, self.up_padding])

    def forward(self, x):
        self.vis.clear()
        skips = []
        out = x
        self.vis.append(out)

        for i in range(self.depth):
            out = self.encoder_layers[i](out)
            skips.append(out)
            out = F.max_pool1d(out, kernel_size=self.down_kernel[i], stride=self.strides[i],padding=self.down_padding[i])
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

        return out

    def visualizer(self):
        for item in self.vis:
            print(item.shape)

if __name__ == "__main__":
    config = {
        "Serie length":         800,
        "Input size":           1,
        "Encoder structure":    [1, 64, 128, 256, 512],
        "Decoder structure":    [512, 256, 128, 64, 100],
        "Kernel size":          [3] * 9,
        "Down kernel":          [7] * 4,
        "Up kernel":            [7] * 4,
        "Stride":               [2, 3, 2, 2],
        "Activate function":    nn.GELU()
    }

    input_tensor = torch.rand((4, config["Input size"], config["Serie length"]))
    model = unet_base_1D(config)
    output = model(input_tensor)
    model.structure_calculate(True)
    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    model.visualizer()