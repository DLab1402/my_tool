import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
    visualize = []
    def __init__(self):
        super(UNet1D, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.out = nn.Conv1d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        self.visualize.clear()
        # Encoder
        e1 = self.enc1(x)
        self.visualize.append(e1)
        # print('layer1: {}'.format(e1.shape))
        e2 = self.enc2(F.max_pool1d(e1, 2))
        self.visualize.append(e2)
        # print('layer2: {}'.format(e2.shape))
        e3 = self.enc3(F.max_pool1d(e2, 2))
        self.visualize.append(e3)
        # print('layer3: {}'.format(e3.shape))
        e4 = self.enc4(F.max_pool1d(e3, 2))
        # print('layer4: {}'.format(e4.shape))
        self.visualize.append(e4)
        # Bottleneck
        b = self.bottleneck(F.max_pool1d(e4, 2))
        # print('layer5: {}'.format(b.shape))
        self.visualize.append(b)
        # Decoder
        d4 = self.upconv4(b)
        # print(d4.shape)
        d4 = torch.cat((d4, e4), dim=1)
        # print(d4.shape)
        d4 = self.dec4(d4)
        # print('layer6: {}'.format(d4.shape))
        self.visualize.append(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        # print('layer7: {}'.format(d3.shape))
        self.visualize.append(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        # print('layer8: {}'.format(d2.shape))
        self.visualize.append(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        # print('layer9: {}'.format(d1.shape))
        self.visualize.append(d1)
        # Output layer
        out = self.out(d1)
        return out

# Example usage
# input_tensor = torch.rand((4, 1, 1024))  # Example input: batch size 1, 1 channel, sequence length 1024
# model = UNet1D()
# output = model(input_tensor)
# print(output.shape)