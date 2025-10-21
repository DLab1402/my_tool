import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_batchnorm(torch.nn.Module):
    """
    A block of Conv1d -> BatchNorm -> Activation
    """
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = 1, activation = 'relu'):
        super().__init__()
        # Use nn.ReLU() as a layer for consistency
        self.activation_str = activation
        self.conv1 = torch.nn.Conv1d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
        self.batchnorm = torch.nn.BatchNorm1d(num_out_filters)
        if self.activation_str == 'relu':
            self.activation = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        if self.activation_str == 'relu':
            return self.activation(x)
        else:
            return x


class Multiresblock(torch.nn.Module):
    """
    MultiRes Block
    """
    def __init__(self, num_in_channels, num_filters, filter_rate = [1.67, 0.333, 0.5], kernel_size=[3,5,7]):
        super().__init__()
        
        filt_cnt_1 = int(num_filters*(filter_rate[1]))
        filt_cnt_2 = int(num_filters*(filter_rate[2]))
        filt_cnt_3 = int(num_filters*(1-(filter_rate[1])-(filter_rate[2])))
        
        self.conv_1 = Conv1d_batchnorm(num_in_channels, filt_cnt_1, kernel_size = kernel_size[0], activation='relu')
        self.conv_2 = Conv1d_batchnorm(filt_cnt_1, filt_cnt_2, kernel_size = kernel_size[1], activation='relu')
        self.conv_3 = Conv1d_batchnorm(filt_cnt_2, filt_cnt_3, kernel_size = kernel_size[2], activation='relu')
        
        self.total_filters = filt_cnt_1 + filt_cnt_2 + filt_cnt_3
        self.bn = torch.nn.BatchNorm1d(self.total_filters)
        
        self.shortcut = Conv1d_batchnorm(num_in_channels, self.total_filters, kernel_size=1, activation='None')

    def forward(self, x):
        res1 = self.conv_1(x)
        res2 = self.conv_2(res1)
        res3 = self.conv_3(res2)
        
        concat = torch.cat([res1,res2,res3],axis=1)
        concat = self.bn(concat)
        
        res_shortcut = self.shortcut(x)
        
        return torch.nn.functional.relu(concat + res_shortcut)


class Respath(torch.nn.Module):
    """
    ResPath
    """
    def __init__(self, num_in_filters, num_out_filters, respath_length=4):
        super().__init__()
        
        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])

        for i in range(respath_length):
            if i == 0:
                self.shortcuts.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=3, activation='relu'))
            else:
                self.shortcuts.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=3, activation='relu'))
                
    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            conv = self.convs[i](x)
            x = torch.nn.functional.relu(shortcut + conv)
        return x


class MultiResUnet(torch.nn.Module):
    """
    MultiResUNet
    """
    def __init__(self, para, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.para = para
        
        en_str, bn, de_str, resp, ups = self.structure_calculate()

        self.encoder_layer = torch.nn.ModuleList([])
        self.decoder_layer = torch.nn.ModuleList([])
        self.respath_layer = torch.nn.ModuleList([])
        self.ups_layer = torch.nn.ModuleList([])
        self.pooling = torch.nn.ModuleList([])

        L = len(en_str)

        for i in range(L):
            self.encoder_layer.append(Multiresblock(en_str[i][0], self.para["Filter nums"][i], kernel_size=self.para["Block kernel"]))
            self.pooling.append(torch.nn.MaxPool1d(kernel_size=self.para["Pooling kernel"][i], stride=2))
            self.respath_layer.append(Respath(resp[i][0], resp[i][1], respath_length=self.para.get("Respath length", 4)))

        self.bottle_neck = Multiresblock(bn[0], self.para["Filter nums"][-1], kernel_size=self.para["Block kernel"])

        for i in range(L):
            self.decoder_layer.append(Multiresblock(de_str[i][0], self.para["Filter nums"][L-i-1], kernel_size=self.para["Block kernel"]))
            self.ups_layer.append(torch.nn.ConvTranspose1d(ups[i][0], ups[i][1], kernel_size=self.para["Transpose kernel"][i], stride=2))

        self.conv_final = Conv1d_batchnorm(de_str[-1][1], self.para["Number class"], kernel_size=1, activation='None')

    def structure_calculate(self):
        # ... (This function is complex but seems logically sound, keeping it as is)
        L = len(self.para["Filter nums"])
        en_str = []
        for i in range(L):
            if i == 0:
                en_str.append([1,self.para["Filter nums"][i]])
            else:
                en_str.append([int(self.para["Filter nums"][i-1]*self.alpha),self.para["Filter nums"][i]])
        bn = [int(self.para["Filter nums"][-1]*self.alpha),self.para["Filter nums"][-1]]
        de_str = []
        for i in range(L):
            if i == 0:
                de_str.append([int(self.para["Filter nums"][-1]*(self.alpha+1)),self.para["Filter nums"][-1]])
            else:
                de_str.append([int(self.para["Filter nums"][L-i]*(self.alpha+1)),self.para["Filter nums"][L-i-1]])
        resp = []
        for i in range(L):
            resp.append([self.para["Filter nums"][i],int(self.para["Filter nums"][i]*self.alpha)])
        ups = []
        for i in range(L):
            ups.append([self.para["Filter nums"][L-i-1],self.para["Filter nums"][L-i-1]])
        return en_str, bn, de_str, resp, ups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A list to hold the outputs from the encoder for skip connections
        encoder_outputs = []
        
        # --- Encoder Path ---
        temp_x = x
        for i in range(len(self.encoder_layer)):
            # Pass through MultiResBlock
            encoder_out = self.encoder_layer[i](temp_x)
            
            # Save the output for the skip connection
            encoder_outputs.append(encoder_out)
            
            # Pass through pooling layer
            temp_x = self.pooling[i](encoder_out)
        
        # --- Bottleneck ---
        bottleneck_out = self.bottle_neck(temp_x)

        # --- Decoder Path ---
        temp_d = bottleneck_out
        for i in range(len(self.decoder_layer)):
            # Upsampling
            upsampled = self.ups_layer[i](temp_d)
            
            # Get the corresponding skip connection from the encoder
            skip_connection = encoder_outputs[-(i + 1)]
            
            # Concatenate skip connection with the upsampled output
            # Ensure they have the same length after upsampling
            if upsampled.shape[-1] != skip_connection.shape[-1]:
                 upsampled = F.interpolate(upsampled, size=skip_connection.shape[-1], mode='linear', align_corners=False)

            concat = torch.cat([upsampled, skip_connection], axis=1)
            
            # Pass through the decoder's MultiResBlock
            temp_d = self.decoder_layer[i](concat)

        # Final convolution to get the desired number of output channels
        out = self.conv_final(temp_d)
        
        # --- CRITICAL FIX: REMOVED SOFTMAX ---
        # F.softmax is for classification. For L1Loss (regression), we need raw output.
        return out
