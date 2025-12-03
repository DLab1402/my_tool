import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_batchnorm(torch.nn.Module):
    """
    Khối Conv1d chuẩn: Conv -> BatchNorm -> LeakyReLU -> (Optional Dropout)
    """
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=1, activation='leaky_relu', dropout_rate=0.0):
        super().__init__()
        self.activation_str = activation
        # Padding 'same' cho stride 1
        padding_val = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else 0
        
        self.conv1 = torch.nn.Conv1d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding=padding_val)
        
        # --- QUAN TRỌNG: Khởi tạo trọng số Kaiming (He Init) ---
        # Giúp mạng sâu hội tụ tốt hơn, tránh vanishing gradient
        if activation in ['relu', 'leaky_relu']:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif activation == 'linear':
            nn.init.xavier_normal_(self.conv1.weight)
        
        self.batchnorm = torch.nn.BatchNorm1d(num_out_filters)
        
        # Dropout nhẹ (nếu cần)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Activation: Ưu tiên LeakyReLU cho tín hiệu sóng
        if self.activation_str == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_str == 'leaky_relu': 
            self.activation = nn.LeakyReLU(0.1, inplace=True) # Slope 0.1 để giữ thông tin âm
        elif self.activation_str == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        
        if self.activation_str != 'None':
            x = self.activation(x)
            
        x = self.dropout(x)
        return x


class Multiresblock(torch.nn.Module):
    """
    MultiRes Block: Đa thang đo với LeakyReLU
    """
    def __init__(self, num_in_channels, num_filters, filter_rate=[1.67, 0.333, 0.5], kernel_size=[3, 5, 7], dropout_rate=0.0):
        super().__init__()
        
        # Tính toán kênh chính xác
        filt_cnt_1 = int(num_filters * filter_rate[1])
        filt_cnt_2 = int(num_filters * filter_rate[2])
        filt_cnt_3 = int(num_filters * (1 - filter_rate[1] - filter_rate[2]))
        
        act_fn = 'leaky_relu' # Dùng LeakyReLU cho toàn bộ block
        
        self.conv_1 = Conv1d_batchnorm(num_in_channels, filt_cnt_1, kernel_size=kernel_size[0], activation=act_fn, dropout_rate=dropout_rate)
        self.conv_2 = Conv1d_batchnorm(filt_cnt_1, filt_cnt_2, kernel_size=kernel_size[1], activation=act_fn, dropout_rate=dropout_rate)
        self.conv_3 = Conv1d_batchnorm(filt_cnt_2, filt_cnt_3, kernel_size=kernel_size[2], activation=act_fn, dropout_rate=dropout_rate)
        
        self.total_filters = filt_cnt_1 + filt_cnt_2 + filt_cnt_3
        self.bn = torch.nn.BatchNorm1d(self.total_filters)
        
        # Shortcut 1x1
        self.shortcut = Conv1d_batchnorm(num_in_channels, self.total_filters, kernel_size=1, activation='None')
        
        self.final_act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res1 = self.conv_1(x)
        res2 = self.conv_2(res1)
        res3 = self.conv_3(res2)
        
        concat = torch.cat([res1, res2, res3], axis=1)
        concat = self.bn(concat)
        
        res_shortcut = self.shortcut(x)
        
        return self.final_act(concat + res_shortcut)


class Respath(torch.nn.Module):
    """
    ResPath: Cầu nối ngữ nghĩa
    """
    def __init__(self, num_in_filters, num_out_filters, respath_length=4):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        
        act_fn = 'leaky_relu'

        for i in range(respath_length):
            if i == 0:
                self.shortcuts.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=3, activation=act_fn))
            else:
                self.shortcuts.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=3, activation=act_fn))
        
        self.final_act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            conv = self.convs[i](x)
            x = self.final_act(shortcut + conv)
        return x


class MultiResUnet(torch.nn.Module):
    """
    MultiResUNet (Optimized Version - Kaiming Init + LeakyReLU)
    """
    def __init__(self, para, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.para = para
        
        self.dropout_rate = self.para.get("dropout_rate", 0.0) # Mặc định 0.0 cho Encoder để giữ thông tin

        en_str, bn_str, de_str, resp_str, ups_str = self.structure_calculate()

        self.encoder_layer = torch.nn.ModuleList([])
        self.decoder_layer = torch.nn.ModuleList([])
        self.respath_layer = torch.nn.ModuleList([])
        self.ups_layer = torch.nn.ModuleList([])
        self.pooling = torch.nn.ModuleList([])

        L = len(en_str)

        # --- Encoder ---
        for i in range(L):
            # Encoder dùng dropout thấp hoặc 0
            self.encoder_layer.append(Multiresblock(en_str[i][0], self.para["Filter nums"][i], kernel_size=self.para["Block kernel"], dropout_rate=self.dropout_rate))
            self.pooling.append(torch.nn.MaxPool1d(kernel_size=self.para["Pooling kernel"][i], stride=2))
            self.respath_layer.append(Respath(resp_str[i][0], resp_str[i][1], respath_length=self.para.get("Respath length", 4)))

        # --- Bottleneck (Nơi quan trọng nhất cần Dropout) ---
        self.bottle_neck = Multiresblock(bn_str[0], self.para["Filter nums"][-1], kernel_size=self.para["Block kernel"])
        self.bottleneck_dropout = nn.Dropout(0.3) # Dropout 30% ở đáy

        # --- Decoder ---
        for i in range(L):
            self.ups_layer.append(torch.nn.ConvTranspose1d(ups_str[i][0], ups_str[i][1], kernel_size=self.para["Transpose kernel"][i], stride=2))
            
            filter_num_idx = L - 1 - i
            self.decoder_layer.append(Multiresblock(de_str[i][0], self.para["Filter nums"][filter_num_idx], kernel_size=self.para["Block kernel"], dropout_rate=self.dropout_rate))

        # --- Final Conv ---
        last_filter_arg = self.para["Filter nums"][0]
        actual_last_out_ch = self.get_block_actual_output(last_filter_arg)
        
        # Output layer: Linear (không activation) để hồi quy giá trị thực
        self.conv_final = Conv1d_batchnorm(actual_last_out_ch, self.para["Number class"], kernel_size=1, activation='None')

    def get_block_actual_output(self, num_filters):
        filter_rate = [1.67, 0.333, 0.5]
        filt_cnt_1 = int(num_filters * filter_rate[1])
        filt_cnt_2 = int(num_filters * filter_rate[2])
        filt_cnt_3 = int(num_filters * (1 - filter_rate[1] - filter_rate[2]))
        return filt_cnt_1 + filt_cnt_2 + filt_cnt_3

    def structure_calculate(self):
        filter_nums = self.para["Filter nums"]
        L = len(filter_nums)
        
        en_str, resp_str, de_str, ups_str = [], [], [], []
        
        current_input_ch = self.para.get("Input size", 1) 
        if isinstance(current_input_ch, list): current_input_ch = 1 

        encoder_actual_outputs = []

        # Encoder Flow
        for i in range(L):
            en_str.append([current_input_ch, filter_nums[i]])
            block_out = self.get_block_actual_output(filter_nums[i])
            encoder_actual_outputs.append(block_out)
            resp_str.append([block_out, block_out])
            current_input_ch = block_out

        # Bottleneck Flow
        bn_input_ch = encoder_actual_outputs[-1]
        bn_filter_arg = filter_nums[-1] 
        bn_str = [bn_input_ch, bn_filter_arg]
        bn_output_ch = self.get_block_actual_output(bn_filter_arg)

        # Decoder Flow
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
        
        # Encoder Pass
        for i in range(len(self.encoder_layer)):
            out = self.encoder_layer[i](temp_x)
            res_out = self.respath_layer[i](out)
            encoder_outputs.append(res_out)
            temp_x = self.pooling[i](out)
        
        # Bottleneck Pass
        bottleneck_out = self.bottle_neck(temp_x)
        bottleneck_out = self.bottleneck_dropout(bottleneck_out) # Apply Dropout here

        # Decoder Pass
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