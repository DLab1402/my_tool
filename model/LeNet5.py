#LeNet Model 
import torch
import torch.nn as nn
import torch.nn.functional as F
#     para = {
#         'input_image_size': (320, 320),
#         'input_channel': 1,
#         'number_of_conv_layer':2,
#         'number_of_fc_layer':2,
#         'num_classes':2,
#     }

class LeNet(nn.Module):
    def __init__(self, para):
        super(LeNet,self).__init__()
        self.para=para
        self.input_image_size=para['input_image_size']
        self.input_channel=para['input_channel']
        self.num_conv_layer=para['number_of_conv_layer']
        self.num_fc_layer=para['number_of_fc_layer']
        self.fc_feateure=para['fc_feature']
        self.conv_channels=para['conv_channels']
        self.Conv_layer=nn.ModuleList()
        self.FC_layer=nn.ModuleList()
        self.build_conv_layer()
        self.build_fc_layer()
    def build_conv_layer(self):
        for i in range(self.num_conv_layer):
            in_channels=self.input_channel if i==0 else self.conv_channels[i-1]
            out_channels=self.conv_channels[i]
            conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=7,stride=1,padding=2),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2))
            self.Conv_layer.append(conv)
    def forward_convs(self, x):
        for conv in self.Conv_layer:
            x = conv(x)
        return x

    def build_fc_layer(self):
        in_features=0
        with torch.no_grad():
            x=torch.randn(1,self.input_channel,self.input_image_size[0],self.input_image_size[1])
            x=self.forward_convs(x)
            in_features=x.flatten(1).shape[1]
        fc=nn.Sequential(nn.Linear(in_features,self.fc_feateure[0]),
                         nn.ReLU(),
                         nn.Linear(self.fc_feateure[0],self.fc_feateure[1]),
                         nn.Softmax(dim=1))
        self.FC_layer.append(fc)
    def forward_fc(self, x):
        x = x.flatten(1)
        for fc in self.FC_layer:
            x = fc(x)
        return x

    def forward(self, x):
        x = self.forward_convs(x)
        x = self.forward_fc(x)
        return x
    def summary(self):
        print("="*40)
        print(f"{'Layer':<15}{'Output Shape':<20}{'Details'}")
        print("="*40)
        for i, conv in enumerate(self.Conv_layer):
            print(f"Conv{i+1:<10} {str(conv):<20}")
        for i, fc in enumerate(self.FC_layer):
            print(f"FC{i+1:<10} {str(fc):<20}")
        print("="*40)

if __name__ == "__main__":
    para = {
        'input_image_size': (608, 608),
        'input_channel': 1,
        'number_of_conv_layer':3,
        'number_of_fc_layer':2,
        'num_classes':2,
        'conv_channels': [6, 16,24],
        'fc_feature':[104,2]
    }

    model = LeNet(para)
    x = torch.randn(1, 1, 608, 608)
    y = model(x)
    print("Output shape:", y.shape)
    print(f"Predicted class:{y}")
    model.summary()