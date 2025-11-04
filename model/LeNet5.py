import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, para):
        super(LeNet, self).__init__()
        self.para = para
        self.Conv_layer = nn.ModuleList()
        self.BatchNorm_layer = nn.ModuleList()
        self.FC_layer = nn.ModuleList()

        self.conv_layer_maker()
        with torch.no_grad():
            x = torch.randn(1, para['input_channel'], 32, 32)
            x = self.forward_convs(x)
            in_features = x.flatten(1).shape[1]
        self.FC_maker(in_features)
    def conv_layer_maker(self):
        for i in range(self.para['number_of_conv_layer']):
            in_ch = self.para['input_channel'] if i == 0 else self.para['output_channel'][i - 1]
            out_ch = self.para['output_channel'][i]
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=self.para['Kernel_size'][i],
                stride=self.para['Stride'][i],
                padding=self.para['Padding'][i]
            )
            self.Conv_layer.append(conv)

            if self.para.get('use_batchnorm', False):
                self.BatchNorm_layer.append(nn.BatchNorm2d(out_ch))
            else:
                self.BatchNorm_layer.append(None)

    def FC_maker(self, in_features):
        for i in range(self.para['number_of_FC_layer']):
            in_f = in_features if i == 0 else self.para['FC_features'][i - 1]
            fc = nn.Linear(in_f, self.para['FC_features'][i])
            self.FC_layer.append(fc)

    def choose_activation(self, name):
        if name == 'ReLU':
            return F.relu
        elif name == 'Sigmoid':
            return torch.sigmoid
        elif name == 'Tanh':
            return torch.tanh
        elif name == 'Softmax':
            return lambda x: F.softmax(x, dim=1)
        elif name == 'None' or name is None:
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def choose_pooling(self, name):
        if name == 'MaxPool':
            return F.max_pool2d
        elif name == 'AvgPool':
            return F.avg_pool2d
        else:
            raise ValueError(f"Unsupported pooling: {name}")

    def forward_convs(self, x):
        for i in range(self.para['number_of_conv_layer']):
            act = self.choose_activation(self.para['Activation_Func'][i])
            pool = self.choose_pooling(self.para['Pooling_type'][i])
            x = self.Conv_layer[i](x)
            if self.para.get('use_batchnorm', False) and self.BatchNorm_layer[i] is not None:
                x = self.BatchNorm_layer[i](x)
            x = act(x)
            x = pool(x, 2, 2)
        return x

    def forward(self, x):
        x = self.forward_convs(x)
        x = torch.flatten(x, 1)
        for i in range(self.para['number_of_FC_layer']):
            x = self.FC_layer[i](x)
            act = self.choose_activation(self.para['FC_Activation_Func'][i])
            x = act(x)
        return x

    def summary(self):
        print("LeNet Model")
        for i in range(self.para['number_of_conv_layer']):
            print(f"Conv Layer {i+1}: {self.Conv_layer[i]}")
            if self.para.get('use_batchnorm', False):
                print(f"    BatchNorm: {self.BatchNorm_layer[i]}")
            print(f"    Activation: {self.para['Activation_Func'][i]}")
            print(f"    Pooling: {self.para['Pooling_type'][i]}(kernel_size=2, stride=2)")
        for i in range(self.para['number_of_FC_layer']):
            print(f"FC Layer {i+1}: {self.FC_layer[i]}")
            print(f"    Activation: {self.para['FC_Activation_Func'][i]}")

if __name__ == "__main__":
    para = {
        'input_channel': 1,
        'number_of_conv_layer': 2,
        'number_of_FC_layer': 2,
        'FC_features': [84, 2],
        'output_channel': [6, 16],
        'Kernel_size': [7, 5],
        'Stride': [1, 1],
        'Padding': [0, 0],
        'Activation_Func': ['ReLU', 'ReLU'],
        'Pooling_type': ['MaxPool', 'MaxPool'],
        'FC_Activation_Func': ['ReLU', 'Softmax'], 
        'use_batchnorm': True,
    }
    model = LeNet(para)
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    model.summary()
