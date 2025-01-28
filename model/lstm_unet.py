import torch
import torch.nn as nn
import torch.nn.functional as F

class lstm_unet(nn.Module):
    
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
            "Activate function":    []
    }

    def __init__(self,para):
        super(lstm_unet, self).__init__()
        self.encoder_layer = nn.ModuleList()
        self.decoder_layer = nn.ModuleList()
        self.upconv_layer = nn.ModuleList()
        self.para = para 
        tem = self.structure_calculate()
        self.conv_e_struc = tem[1]
        self.bn_struc = tem[2]
        self.conv_d_struc = tem[3]
        self.down_struc = tem[4]
        self.up_struc = tem[5]

        self.lstm = self.lstm_block()
        
        self.encoder_block_maker()
        self.bottleneck = self.conv_block(self.conv_struct_format(self.bn_struc[0][0],self.bn_struc[0][1]),self.bn_struc[1],self.bn_struc[2])
        self.decoder_block_maker()
    
    def structure_calculate(self,visualize=False):
        padding_conv_cal = lambda Lin,Lout,kernel,stride: 0.5*((Lout-1)*stride-Lin+kernel)
        padding_convT_cal = lambda Lin,Lout,kernel,stride: 0.5*((Lin-1)*stride-Lout+kernel)
        conv_e_struc = []
        conv_d_struc = []
        bottle_neck = [self.para["Encoder structure"][-1],self.para["Decoder structure"][0]]
        padding_conv = []
        padding_encoder = []
        padding_decoder = []
        
        Lout = [self.para["Serie length"]]
        L = len(self.para["Encoder structure"])
        for i in range(len(self.para["Stride"])):
            Lout.append(int(Lout[i]/self.para["Stride"][i]))

        for i in range(L-1):
            conv_e_struc.append([self.para["Encoder structure"][i],self.para["Encoder structure"][i+1]])
            conv_d_struc.append([self.para["Encoder structure"][i+1]+self.para["Decoder structure"][L-i-2],self.para["Decoder structure"][L-i-1]])

            padding_encoder.append(padding_conv_cal(Lout[i],Lout[i+1],self.para["Kernel encoder"][i],self.para["Stride"][i]))
            padding_decoder.append(padding_convT_cal(Lout[i+1],Lout[i],self.para["Kernel decoder"][i],self.para["Stride"][i]))

        for i in range(len(self.para["Kernel size"])):
            padding_conv.append((self.para["Kernel size"][i]-1)/2)

        if visualize == True:
            print(Lout)
            print([conv_e_struc,self.para["Kernel size"][0:L-1],padding_conv[0:L-1]])
            print([bottle_neck, self.para["Kernel size"][L-1],padding_conv[L-1]])
            print([list(reversed(conv_d_struc)),self.para["Kernel size"][L:2*L-1],padding_conv[L:2*L-1]])
            print([self.para["Kernel encoder"],self.para["Stride"],padding_encoder])
            print([self.para["Kernel decoder"],list(reversed(self.para["Stride"])),list(reversed(padding_decoder))])

        return [Lout,
            [conv_e_struc,self.para["Kernel size"][0:L-1],padding_conv[0:L-1]],
            [bottle_neck, self.para["Kernel size"][L-1],padding_conv[L-1]],
            [list(reversed(conv_d_struc)),self.para["Kernel size"][L:2*L-1],padding_conv[L:2*L-1]],
            [self.para["Kernel encoder"],self.para["Stride"],padding_encoder],
            [self.para["Kernel decoder"],list(reversed(self.para["Stride"])),list(reversed(padding_decoder))]]
    
    def lstm_block(self):
        input_size = self.para["Input size"]
        hidden_size = self.para["Hidden size"]
        num_layers = self.para["Number layers"]
        return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def conv_struct_format(self,input,ouput):
        conv_structure = [input]
        for i in range(4):
            conv_structure.append(ouput)
        return conv_structure

    def conv_block(self,layer_dims,kernel_size,padding):
        conv = []
        fnc_act = self.para["Activate function"]
        for i in range(len(layer_dims)-1):
            conv.append(nn.BatchNorm1d(layer_dims[i]))
            conv.append(nn.Conv1d(layer_dims[i], layer_dims[i + 1],kernel_size=kernel_size, padding=int(padding)))
            conv.append(fnc_act)
        return nn.Sequential(*conv)
    
    def encoder_block_maker(self):
        layer_dims = self.conv_e_struc[0]
        kernel_size = self.conv_e_struc[1]
        padding = self.conv_e_struc[2]
        for i in range(len(self.conv_e_struc[0])):
            self.encoder_layer.append(self.conv_block(self.conv_struct_format(layer_dims[i][0],layer_dims[i][1]),kernel_size[i],int(padding[i])))

    def decoder_block_maker(self):
        layer_dims = self.conv_d_struc[0]
        kernel_size = self.conv_d_struc[1]
        padding = self.conv_d_struc[2]

        up_ker = self.up_struc[0]
        up_str = self.up_struc[1]
        up_pad = self.up_struc[2]

        for i in range(len(self.conv_d_struc[0])):
            self.upconv_layer.append(nn.ConvTranspose1d(self.para["Decoder structure"][i], self.para["Decoder structure"][i], kernel_size=up_ker[i], stride=up_str[i], padding=int(up_pad[i])))
            self.decoder_layer.append(self.conv_block(self.conv_struct_format(layer_dims[i][0],layer_dims[i][1]),kernel_size[i],int(padding[i])))
        
    def forward(self, x):
        L = len(self.conv_e_struc[0])
        e = []
        self.vis.clear()
        out = x.transpose(1, 2)
        out,_ = self.lstm(out)
        out = out.transpose(1, 2)
        self.vis.append(out)
        for i,layer in enumerate(self.encoder_layer):
            # if i == 0:
            #     out = layer(x)
            # else:
            #     out = layer(out)
            out = layer(out)
            e.append(out)
            out = F.max_pool1d(out, kernel_size=self.down_struc[0][i], stride=self.down_struc[1][i],padding=int(self.down_struc[2][i]))
            self.vis.append(out)
        
        out = self.bottleneck(out)
        self.vis.append(out)

        for i in range(L):
            out = self.upconv_layer[i](out)
            out = torch.cat((out, e[L-i-1]), dim=1)
            out = self.decoder_layer[i](out)
            self.vis.append(out)

        return out
    
    def visualizer(self):
        for item in self.vis:
            print(item.shape)

# Example usage
if __name__ =="__main__":
    para = {"Serie length":         1024,
            "Input size":           1,
            "Hidden size":          100,
            "Number layers":        2,
            "Encoder structure":    [100,64,128,256,512], 
            "Decoder structure":    [512,256,128,64,1],
            "Kernel size":          [3,3,3,3,3,3,3,3,3],
            "Kernel encoder":       [4,4,4,4],
            "Kernel decoder":       [2,2,2,2],
            "Stride":               [2,2,2,2],
            "Activate function":    nn.Tanh()
    }

    input_tensor = torch.rand((4, 1, 1024))
    model = lstm_unet(para)
    model.structure_calculate(True)
    output = model(input_tensor)
    model.visualizer()