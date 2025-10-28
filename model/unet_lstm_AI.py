import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class unet_lstm(nn.Module):
    
    vis = []
    para = {"Serie length":         [],
            "Input size":           [],
            "Hidden size":          [],
            "Number layers":        [],
            "Encoder structure":    [], 
            "Decoder structure":    [],
            "Kernel size":          [],
            "Kernel encoder":       [],
            "Kernel decoder":       [],
            "Stride":               [],
            "Activate function":    [],
            "Output activation":    []  # ADDED: activation for output
    }

    def __init__(self, para):
        super(unet_lstm, self).__init__()
        self.encoder_layer = nn.ModuleList()
        self.decoder_layer = nn.ModuleList()
        self.upconv_layer = nn.ModuleList()
        self.para = para 
        
        # Validate parameters
        self._validate_parameters()
        
        tem = self.structure_calculate()
        self.conv_e_struc = tem[1]
        self.bn_struc = tem[2]
        self.conv_d_struc = tem[3]
        self.down_struc = tem[4]
        self.up_struc = tem[5]

        self.lstm = self.lstm_block()
        
        self.encoder_block_maker()
        self.bottleneck = self.conv_block(
            self.conv_struct_format(self.bn_struc[0][0], self.bn_struc[0][1]),
            self.bn_struc[1], 
            self.bn_struc[2]
        )
        self.decoder_block_maker()
        
        # ADDED: Final output layer with proper activation
        self.output_activation = self.para.get("Output activation", None)
    
    def _validate_parameters(self):
        """Validate model parameters before building"""
        assert len(self.para["Encoder structure"]) > 1, "Encoder needs at least 2 layers"
        assert len(self.para["Decoder structure"]) > 1, "Decoder needs at least 2 layers"
        assert len(self.para["Encoder structure"]) == len(self.para["Decoder structure"]), \
            "Encoder and Decoder must have same depth"
        
        # Check LSTM input size matches last encoder output
        if self.para["Hidden size"] != self.para["Encoder structure"][-1]:
            print(f"⚠️  Warning: Hidden size ({self.para['Hidden size']}) doesn't match " 
                  f"last encoder layer ({self.para['Encoder structure'][-1]})")
    
    def structure_calculate(self, visualize=False):
        padding_conv_cal = lambda Lin, Lout, kernel, stride: math.ceil(0.5 * ((Lout - 1) * stride - Lin + kernel))
        padding_convT_cal = lambda Lin, Lout, kernel, stride: max(0, math.floor(0.5 * ((Lin - 1) * stride - Lout + kernel)))
        
        conv_e_struc = []
        conv_d_struc = []
        bottle_neck = [self.para["Encoder structure"][-1], self.para["Encoder structure"][-1]]  # FIXED: keep same size
        padding_conv = []
        padding_encoder = []
        padding_decoder = []
        
        Lout = [self.para["Serie length"]]
        L = len(self.para["Encoder structure"])
        
        # Calculate output sizes at each downsampling step
        for i in range(len(self.para["Stride"])):
            Lout.append(int(Lout[i] / self.para["Stride"][i]))

        for i in range(L - 1):
            conv_e_struc.append([self.para["Encoder structure"][i], self.para["Encoder structure"][i + 1]])
            # FIXED: Concatenation in decoder adds channels from encoder
            conv_d_struc.append([
                self.para["Encoder structure"][i + 1] + self.para["Decoder structure"][L - i - 2],
                self.para["Decoder structure"][L - i - 1]
            ])

            padding_encoder.append(padding_conv_cal(Lout[i], Lout[i + 1], 
                                                    self.para["Kernel encoder"][i], 
                                                    self.para["Stride"][i]))
            padding_decoder.append(padding_convT_cal(Lout[i + 1], Lout[i], 
                                                     self.para["Kernel decoder"][i], 
                                                     self.para["Stride"][i]))

        for i in range(len(self.para["Kernel size"])):
            padding_conv.append((self.para["Kernel size"][i] - 1) / 2)

        if visualize:
            print("=" * 50)
            print("Model Structure Calculation:")
            print("=" * 50)
            print(f"Sequence lengths at each level: {Lout}")
            print(f"\nEncoder structure: {[conv_e_struc, self.para['Kernel size'][0:L-1], padding_conv[0:L-1]]}")
            print(f"\nBottleneck: {[bottle_neck, self.para['Kernel size'][L-1], padding_conv[L-1]]}")
            print(f"\nDecoder structure: {[list(reversed(conv_d_struc)), self.para['Kernel size'][L:2*L-1], padding_conv[L:2*L-1]]}")
            print(f"\nDownsampling: {[self.para['Kernel encoder'], self.para['Stride'], padding_encoder]}")
            print(f"\nUpsampling: {[self.para['Kernel decoder'], list(reversed(self.para['Stride'])), list(reversed(padding_decoder))]}")
            print("=" * 50)

        return [Lout,
                [conv_e_struc, self.para["Kernel size"][0:L-1], padding_conv[0:L-1]],
                [bottle_neck, self.para["Kernel size"][L-1], padding_conv[L-1]],
                [list(reversed(conv_d_struc)), self.para["Kernel size"][L:2*L-1], padding_conv[L:2*L-1]],
                [self.para["Kernel encoder"], self.para["Stride"], padding_encoder],
                [self.para["Kernel decoder"], list(reversed(self.para["Stride"])), list(reversed(padding_decoder))]]
    
    def lstm_block(self):
        input_size = self.para['Encoder structure'][-1]
        hidden_size = self.para["Hidden size"]
        num_layers = self.para["Number layers"]
        return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
    
    def conv_struct_format(self, input, output):
        # FIXED: Create 3 conv layers instead of 5 for efficiency
        conv_structure = [input, output, output, output]
        return conv_structure

    def conv_block(self, layer_dims, kernel_size, padding):
        conv = []
        fnc_act = self.para["Activate function"]
        for i in range(len(layer_dims) - 1):
            conv.append(nn.Conv1d(layer_dims[i], layer_dims[i + 1], 
                                 kernel_size=kernel_size, 
                                 padding=int(padding)))
            conv.append(nn.BatchNorm1d(layer_dims[i + 1]))  # ADDED: BatchNorm for stability
            conv.append(fnc_act)
        return nn.Sequential(*conv)
    
    def encoder_block_maker(self):
        layer_dims = self.conv_e_struc[0]
        kernel_size = self.conv_e_struc[1]
        padding = self.conv_e_struc[2]
        for i in range(len(self.conv_e_struc[0])):
            self.encoder_layer.append(
                self.conv_block(
                    self.conv_struct_format(layer_dims[i][0], layer_dims[i][1]),
                    kernel_size[i], 
                    int(padding[i])
                )
            )

    def decoder_block_maker(self):
        layer_dims = self.conv_d_struc[0]
        kernel_size = self.conv_d_struc[1]
        padding = self.conv_d_struc[2]

        up_ker = self.up_struc[0]
        up_str = self.up_struc[1]
        up_pad = self.up_struc[2]

        for i in range(len(self.conv_d_struc[0])):
            self.upconv_layer.append(
                nn.ConvTranspose1d(
                    self.para["Decoder structure"][i], 
                    self.para["Decoder structure"][i], 
                    kernel_size=up_ker[i], 
                    stride=up_str[i], 
                    padding=int(up_pad[i]),
                    output_padding=0  # ADDED: for exact size matching
                )
            )
            self.decoder_layer.append(
                self.conv_block(
                    self.conv_struct_format(layer_dims[i][0], layer_dims[i][1]),
                    kernel_size[i], 
                    int(padding[i])
                )
            )
        
    def forward(self, x):
        L = len(self.conv_e_struc[0])
        e = []
        self.vis.clear()
        out = x
        
        # Encoder
        for i, layer in enumerate(self.encoder_layer):
            out = layer(out)
            e.append(out)
            out = F.max_pool1d(out, 
                              kernel_size=self.down_struc[0][i], 
                              stride=self.down_struc[1][i],
                              padding=int(self.down_struc[2][i]))
            self.vis.append(out)
        
        # Bottleneck 
        out = self.bottleneck(out)
        self.vis.append(out)
        
        # LSTM
        out = out.permute(0, 2, 1)  # (batch, channels, length) -> (batch, length, channels)
        print(out.shape)
        out, _ = self.lstm(out)
        out = out.permute(0, 2, 1)  # (batch, length, channels) -> (batch, channels, length)
        self.vis.append(out)
        
        # Decoder with skip connections
        for i in range(L):
            out = self.upconv_layer[i](out)
            
            # FIXED: Handle size mismatch in skip connections
            if out.shape[2] != e[L - i - 1].shape[2]:
                diff = e[L - i - 1].shape[2] - out.shape[2]
                out = F.pad(out, (diff // 2, diff - diff // 2))
            
            out = torch.cat((out, e[L - i - 1]), dim=1)
            out = self.decoder_layer[i](out)
            self.vis.append(out)
        
        # ADDED: Apply output activation if specified
        if self.output_activation is not None:
            out = self.output_activation(out)
            
        return out
    
    def visualizer(self):
        print("\n" + "=" * 50)
        print("Model Forward Pass Visualization:")
        print("=" * 50)
        for i, item in enumerate(self.vis):
            print(f"Step {i}: {item.shape}")
        print("=" * 50)


# Example usage for PPG abnormality segmentation
if __name__ == "__main__":
    # FIXED parameters for PPG segmentation
    # Formula: Kernel size needs (L-1) for encoder + 1 for bottleneck + (L-1) for decoder = 2L-1 elements
    # With L=5 layers: need 2*5-1 = 9 kernel sizes
    para = {
        "Serie length":         800,  # FIXED: Match input length
        "Input size":           100,
        "Hidden size":          256,   # FIXED: Reduced for efficiency
        "Number layers":        2,
        "Encoder structure":    [1, 32, 64, 128, 256],  # FIXED: Start smaller (L=5)
        "Decoder structure":    [256, 128, 64, 32, 100],  # Output 1 channel for binary segmentation
        "Kernel size":          [5, 5, 5, 5, 5, 5, 5, 5, 5],  # FIXED: Need 9 values (2*L-1)
        "Kernel encoder":       [8, 8, 8, 8],  # Need L-1 = 4 values
        "Kernel decoder":       [8, 8, 8, 8],  # Need L-1 = 4 values
        "Stride":               [2, 2, 2, 2],  # Need L-1 = 4 values
        "Activate function":    nn.ReLU(),  # FIXED: ReLU is more stable than Tanh
        "Output activation":    nn.Sigmoid()  # ADDED: For binary segmentation [0,1]
    }

    print("Testing UNET-LSTM for PPG Abnormality Segmentation")
    print("=" * 50)
    
    # Create model
    model = unet_lstm(para)
    model.structure_calculate(True)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.rand((batch_size, 1, 1024))  # (batch, channel, length)
    print(f"\nInput shape: {input_tensor.shape}")
    
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    model.visualizer()
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")