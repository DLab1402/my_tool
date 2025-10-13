import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    para={
        'input_channel':1,
        'number_of_conv_layer':2,
        'number_of_FC_layer':2,
        'FC_features':list[int],
        'output_channel':list[int],
        'Kernel_size':list[int],
        'Stride':list[int],
        'Padding':list[int],
        'Activation_Func':list[str],
        'Pooling_type':list[str],
    }
    def __init__(self,para):
        super(LeNet,self).__init__()
        self.para=para
        self.Conv_layer=nn.ModuleList()
        self.FC_layer=nn.ModuleList()
    def conv_layer_maker(self):
        for i in range(self.para['number_of_conv_layer']):
            if i==0:
                conv=nn.Conv2d(in_channels=self.para['input_channel'],
                               out_channels=self.para['output_channel'][i],
                               kernel_size=self.para['Kernel_size'][i],
                               stride=self.para['Stride'][i],
                               padding=self.para['Padding'][i])
            else:
                conv=nn.Conv2d(in_channels=self.para['output_channel'][i-1],
                               out_channels=self.para['output_channel'][i],
                               kernel_size=self.para['Kernel_size'][i],
                               stride=self.para['Stride'][i],
                               padding=self.para['Padding'][i])
            self.Conv_layer.append(conv)
    def FC_maker(self,in_features):
        for i in range(self.para['number_of_FC_layer']):
            if i==0:
                fc=nn.Linear(in_features=in_features,
                             out_features=self.para['FC_features'][i])
            else:
                fc=nn.Linear(in_features=self.para['FC_features'][i-1],
                             out_features=self.para['FC_features'][i])
            self.FC_layer.append(fc)
    def choose_activation(self,para):
        for i in range(self.para['number_of_conv_layer']):
            if para['Activation_Func'][i]=='ReLU':
                return F.relu
            elif para['Activation_Func'][i]=='Sigmoid':
                return torch.sigmoid
            elif para['Activation_Func'][i]=='Tanh':
                return torch.tanh
            else:
                raise ValueError("Unsupported activation function")
    def choose_pooling(self,para):
        for i in range(self.para['number_of_conv_layer']):
            if para['Pooling_type'][i]=='MaxPool':
                return F.max_pool2d
            elif para['Pooling_type'][i]=='AvgPool':
                return F.avg_pool2d
            else:
                raise ValueError("Unsupported pooling type")
    def forward(self,x):
        self.conv_layer_maker()
        activation_funcs=[self.choose_activation(self.para) for _ in range(self.para['number_of_conv_layer'])]
        pooling_funcs=[self.choose_pooling(self.para) for _ in range(self.para['number_of_conv_layer'])]
        for i in range(self.para['number_of_conv_layer']):
            x=self.Conv_layer[i](x)
            x=activation_funcs[i](x)
            x=pooling_funcs[i](x,2,2)
        x=torch.flatten(x,1)
        in_features=x.shape[1]
        self.FC_maker(in_features)
        for i in range(self.para['number_of_FC_layer']):
            x=self.FC_layer[i](x)
            if i!=self.para['number_of_FC_layer']-1:
                x=F.relu(x)
        x=nn.Softmax(dim=1)(x)
        return x
if __name__=="__main__":
    para={
        'input_channel':1,
        'number_of_conv_layer':2,
        'number_of_FC_layer':2,
        'FC_features':[84,2],
        'output_channel':[6,16],
        'Kernel_size':[7,5],
        'Stride':[1,1],
        'Padding':[0,0],
        'Activation_Func':['ReLU','ReLU'],
        'Pooling_type':['MaxPool','MaxPool'],
    }
    model=LeNet(para)
    x=torch.randn(1,1,32,32)
    y=model(x)
    print(y.shape)
    print(model)
