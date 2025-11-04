import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Generator =====

class Generator(nn.Module):
     # dimension of latent variable
    zDim = 1200
    # kernal size for short range filter
    nkShort = 5
    # number of filters for short range filter
    ncShort = 80
    # kernal size for middle range filter
    nkMiddle = 21
    ncMiddle = 40
    # kernal size for long range filter
    nkLong = 61
    ncLong = 20

    ncTotal = 50
    nkTotal = 15

    ncCONV1 = 50
    # upsampling factor
    nStride = 1
    # number of conditioning variables:  AF/NON-AF label, mean heart rate, standard deviation of heart rate
    nCondVar = 1
    def __init__(self):
        super(Generator, self).__init__()
        # input is Z, going into a convolution
        self.sFilter = nn.Sequential(nn.Conv1d(1, self.ncShort, self.nkShort, self.nStride, padding=2, bias=False),
                                     nn.BatchNorm1d(self.ncShort),
                                     nn.ReLU(inplace=True)
                                     )

        self.mFilter = nn.Sequential(nn.Conv1d(1, self.ncMiddle, self.nkMiddle, self.nStride, padding=10, bias=False),
                                     nn.BatchNorm1d(self.ncMiddle),
                                     nn.ReLU(inplace=True)
                                     )

        self.lFilter = nn.Sequential(nn.Conv1d(1, self.ncLong, self.nkLong, self.nStride, padding=30, bias=False),
                                     nn.BatchNorm1d(self.ncLong),
                                     nn.ReLU(inplace=True)
                                     )

        self.tFilter = nn.Sequential(
            nn.Conv1d(self.ncShort + self.ncMiddle + self.ncLong, self.ncTotal, self.nkTotal, self.nStride, padding=7,
                      bias=False),
            nn.BatchNorm1d(self.ncTotal),
            nn.ReLU(inplace=True)
            )

        self.conv1by11 = nn.Sequential(nn.Conv1d(self.ncTotal, self.ncCONV1, 4, padding=1, bias=False), nn.Tanh())

        self.conv1by1 = nn.Sequential(nn.Conv1d(self.ncCONV1, 1, 2, padding=1, bias=False), nn.Tanh())
    
    def forward(self, x):
        b_size = x.size(0)
        x = x.view(b_size,1,self.zDim)
        xs = self.sFilter(x)
        xm = self.mFilter(x)
        xl = self.lFilter(x)

        x = torch.cat((xs, xm, xl), 1)
        x = self.tFilter(x)
        x = self.conv1by11(x)
        x = self.conv1by1(x)

        x = x.view(b_size, -1)
        min_v = torch.min(x, dim=1)[0].view(b_size, 1)
        range_v = torch.max(x, dim=1)[0].view(b_size, 1) - min_v
        normalised_x = (x - min_v) / range_v
        x = normalised_x.view(b_size, 1, -1)

        return x    


# ===== Discriminator =====
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.para = para
        dis_in = 1
        dis_out = 64
        # architecture
        self.model = nn.Sequential(
            # input 1200
            nn.Conv1d(dis_in, dis_out, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 600
            nn.Conv1d(dis_out, dis_out * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dis_out * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size 300
            nn.Conv1d(dis_out * 2, dis_out * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dis_out * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size 150
            nn.Conv1d(dis_out * 4, dis_out * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm1d(dis_out * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size 72
            nn.Conv1d(dis_out * 8, 1, kernel_size = 8, stride = 1, padding = 0, bias = False),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.model(x)        # [B, 1, L]
        x = self.pool(x)         # [B, 1, 1]
        x = self.fc(x)           # [B, 2]
        return x

if __name__ == "__main__":
    aNet = Discriminator()
    input = torch.randn(2,1,1200)    
    output = aNet(input)
    print(output.size())  # torch.Size([2, 1, 1200])
    bNet = Generator()
    input2 = torch.randn(2,1200,1)
    output2 = bNet(input2)
    print(output2.size())