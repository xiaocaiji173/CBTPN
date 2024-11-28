import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ParameterEncoderProcess

def conv_layer0(chann_in, chann_out, k_size,stride=1, p_size=1):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=stride,padding=p_size),
        # nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def conv_layer(chann_in, chann_out, k_size, stride,p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=stride,padding=p_size),
        nn.BatchNorm2d(chann_out,track_running_stats=True,affine=True),
        # tnn.ReLU()
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layer

def deconv_block(ch_in, ch_out, kernel_size=4, stride=2, padding=1):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out,track_running_stats=True,affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        return deconv


# D()
class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64,indim = 7):
        super(discriminator, self).__init__()

        outputdim0 = 64
        self.para_encoder = ParameterEncoderProcess(7, 64, outputdim0)

        self.fc00 = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2))
        self.fc01 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.2))

        self.layer0_1 = deconv_block(64, d // 2, 4, 4, 0)  # 4
        self.layer0_2 = deconv_block(d // 2, d * 1, 4, 4, 0)  # 16
        self.layer0_3 = deconv_block(d * 1, d * 1, 4, 4, 0)  # 32
        #128*3
        self.layer1 = conv_layer0(3,d,3,1,1)
        self.layer2 = conv_layer(d, d*2, 3, 2, 1) #128
        self.layer3 = conv_layer(d*2, d*2, 3, 1, 1)
        self.layer4 = conv_layer(d*2, d*2, 3, 2, 1) #64
        self.layer5 = conv_layer(d*3, d*4, 3, 1, 1)
        self.layer6 = conv_layer(d*4, d*4, 3, 2, 1) #32
        self.layer7 = conv_layer(d*4, d*2, 3, 1, 1)
        self.layer8 = conv_layer(d*2, d*1, 3, 2, 1) #16
        self.layer9 = conv_layer0(d*1, 4, 3, 2, 1) #8

        self.fc =nn.Sequential(
            nn.Linear(4 * 8 * 8, 1),
            nn.Sigmoid())

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):

        feature_P = self.para_encoder(label[0])
        feature_T = self.fc00(label[1])
        y = self.fc01(torch.cat((feature_P, feature_T), 1))

        label = y.view(len(y), -1, 1, 1)
        y = self.layer0_1(label)
        y = self.layer0_2(y)
        y = self.layer0_3(y)

        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x40 = self.layer4(x3)
        x4 = torch.cat([x40,y],dim = 1)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)

        x9 = x9.view(-1,4*8*8)
        x10 = self.fc(x9)

        return x10, x1,x40,x6,x8

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()