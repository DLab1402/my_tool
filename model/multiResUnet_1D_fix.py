import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_batchnorm(torch.nn.Module):
    """
    A block of Conv1d -> BatchNorm -> Activation
    """
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = 1, activation = 'relu'):
        super().__init__()
        self.activation_str = activation
        # Padding calculation to keep same shape (assuming stride 1)
        padding_val = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else 0
        self.conv1 = torch.nn.Conv1d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding=padding_val)
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
        
        # Tính toán số filter con (int rounding)
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
    MultiResUNet (Fixed 31/32 Channel Mismatch)
    """
    def __init__(self, para, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.para = para
        
        # Tính toán cấu trúc
        en_str, bn_str, de_str, resp_str, ups_str = self.structure_calculate()

        self.encoder_layer = torch.nn.ModuleList([])
        self.decoder_layer = torch.nn.ModuleList([])
        self.respath_layer = torch.nn.ModuleList([])
        self.ups_layer = torch.nn.ModuleList([])
        self.pooling = torch.nn.ModuleList([])

        L = len(en_str)

        # --- Encoder ---
        for i in range(L):
            self.encoder_layer.append(Multiresblock(en_str[i][0], self.para["Filter nums"][i], kernel_size=self.para["Block kernel"]))
            self.pooling.append(torch.nn.MaxPool1d(kernel_size=self.para["Pooling kernel"][i], stride=2))
            self.respath_layer.append(Respath(resp_str[i][0], resp_str[i][1], respath_length=self.para.get("Respath length", 4)))

        # --- Bottleneck ---
        self.bottle_neck = Multiresblock(bn_str[0], self.para["Filter nums"][-1], kernel_size=self.para["Block kernel"])

        # --- Decoder ---
        for i in range(L):
            self.ups_layer.append(torch.nn.ConvTranspose1d(ups_str[i][0], ups_str[i][1], kernel_size=self.para["Transpose kernel"][i], stride=2))
            
            filter_num_idx = L - 1 - i
            self.decoder_layer.append(Multiresblock(de_str[i][0], self.para["Filter nums"][filter_num_idx], kernel_size=self.para["Block kernel"]))

        # --- Final Conv ---
        # SỬA LỖI Ở ĐÂY: Input channels là output thực tế của decoder cuối
        last_filter_arg = self.para["Filter nums"][0]
        actual_last_out_ch = self.get_block_actual_output(last_filter_arg) # Tính toán ra 31
        
        self.conv_final = Conv1d_batchnorm(actual_last_out_ch, self.para["Number class"], kernel_size=1, activation='None')

    def get_block_actual_output(self, num_filters):
        """Hàm tính số kênh thực tế (khớp với Multiresblock)"""
        filter_rate = [1.67, 0.333, 0.5]
        filt_cnt_1 = int(num_filters * filter_rate[1])
        filt_cnt_2 = int(num_filters * filter_rate[2])
        filt_cnt_3 = int(num_filters * (1 - filter_rate[1] - filter_rate[2]))
        return filt_cnt_1 + filt_cnt_2 + filt_cnt_3

    def structure_calculate(self):
        filter_nums = self.para["Filter nums"]
        L = len(filter_nums)
        
        en_str, resp_str, de_str, ups_str = [], [], [], []
        
        # Lấy input size
        current_input_ch = self.para.get("Input size", 1) 
        if isinstance(current_input_ch, list): current_input_ch = 1 

        encoder_actual_outputs = []

        # 1. Encoder
        for i in range(L):
            en_str.append([current_input_ch, filter_nums[i]])
            
            # Tính output thực tế
            block_out = self.get_block_actual_output(filter_nums[i])
            encoder_actual_outputs.append(block_out)
            
            resp_str.append([block_out, block_out])
            current_input_ch = block_out

        # 2. Bottleneck
        bn_input_ch = encoder_actual_outputs[-1]
        bn_filter_arg = filter_nums[-1] 
        bn_str = [bn_input_ch, bn_filter_arg]
        bn_output_ch = self.get_block_actual_output(bn_filter_arg)

        # 3. Decoder
        current_dec_input = bn_output_ch
        for i in range(L):
            idx = L - 1 - i
            skip_ch = resp_str[idx][1]
            ups_out_ch = skip_ch 
            
            ups_str.append([current_dec_input, ups_out_ch])
            dec_in_ch = ups_out_ch + skip_ch
            de_str.append([dec_in_ch, filter_nums[idx]])
            
            current_dec_input = self.get_block_actual_output(filter_nums[idx])

        return en_str, bn_str, de_str, resp_str, ups_str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        temp_x = x
        
        # Encoder
        for i in range(len(self.encoder_layer)):
            out = self.encoder_layer[i](temp_x)
            res_out = self.respath_layer[i](out)
            encoder_outputs.append(res_out)
            temp_x = self.pooling[i](out)
        
        # Bottleneck
        bottleneck_out = self.bottle_neck(temp_x)

        # Decoder
        temp_d = bottleneck_out
        for i in range(len(self.decoder_layer)):
            upsampled = self.ups_layer[i](temp_d)
            skip_connection = encoder_outputs[-(i + 1)]
            
            if upsampled.shape[-1] != skip_connection.shape[-1]:
                upsampled = F.interpolate(upsampled, size=skip_connection.shape[-1], mode='linear', align_corners=False)

            concat = torch.cat([upsampled, skip_connection], axis=1)
            temp_d = self.decoder_layer[i](concat)

        out = self.conv_final(temp_d)
        return out