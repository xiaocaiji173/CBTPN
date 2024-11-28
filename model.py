import torch
import torch.nn as nn
import torch.nn.functional as F
# from SAtten import *

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class G_de_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1):
        super(G_de_block, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out

class G_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=2, padding=1):
        super(G_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ParameterEncoderProcess(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParameterEncoderProcess, self).__init__()
        # self.shared_layer = nn.Linear(input_dim + 1, hidden_dim)  # Adjusted input_dim to include the additional feature
        self.shared_layer = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.heatlayer1= nn.Linear(hidden_dim*2, output_dim)
        self.heatlayer2 = nn.Linear(hidden_dim * 2, output_dim)
        self.heatlayer3 = nn.Linear(hidden_dim * 3, output_dim)
        self.output_layer = nn.Linear(hidden_dim*2, output_dim)
        self.originfeaturelayer = nn.Linear(input_dim, hidden_dim)
        self.t_feature = nn.Linear(1, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=4, num_encoder_layers=1, dim_feedforward=128)

    def forward(self, x):
        #  origin feature
        x_origin = F.relu(self.originfeaturelayer(x))
        # x_t_feature: Use only the second dimension's last value
        x_t_input = x[:, -1:] # Extract last value of the second dimension
        x_t_feature = F.relu(self.t_feature(x_t_input))
        # Modify input: add a new feature (second element's fourth power and square root of the fourth element)
        new_feature = x[:, 1:2] ** 4  # Second element's fourth power
        sqrt_feature = torch.sqrt(x[:, 3:4])  # Square root of the fourth element
        x = torch.cat([x[:, :2], new_feature, sqrt_feature, x[:, 3:]], dim=1)  # Insert new features

        # Shared encoding
        # x_shared = F.relu(self.shared_layer(x))
        x_shared = F.relu(self.shared_layer(x.unsqueeze(1))).permute(0, 2, 1)
        features1 = F.relu(self.heatlayer1(torch.cat([x_shared[:,0,:],x_shared[:,1,:]], dim=1)))
        features2 = F.relu(self.heatlayer2(torch.cat([x_shared[:,2,:],x_shared[:,3,:]], dim=1)))
        features3 = F.relu(self.heatlayer3(torch.cat([x_shared[:,4,:],x_shared[:,5,:],x_shared[:,6,:]], dim=1)))
        # features = [features1,features2,features3]
        # concatenated = torch.cat(features, dim=1)
        concatenated = features1+features2+features3

        # Apply transformer
        concatenated = concatenated.unsqueeze(0)  # Add batch dimension for transformer
        x_t_feature = x_t_feature.unsqueeze(0)  # Add batch dimension for transformer
        # print(concatenated.size(),x_t_feature.size())
        transformer_output = self.transformer(concatenated, x_t_feature)
        transformer_output = transformer_output.squeeze(0)  # Remove batch dimension after transformer

        # Combine with x_origin and process final output
        combined = torch.cat([transformer_output, x_origin], dim=1)
        output = F.relu(self.output_layer(combined))
        return output

class GAN_img_to_img(nn.Module):
    def __init__(self, inputdim=7,d = 64,outputdim = 2):
        # BaseModel.__init__(self)
        super(GAN_img_to_img, self).__init__()
        self.fc00 = nn.Sequential(
            nn.Linear(7, 64),
            nn.LeakyReLU(0.2)
        )
        self.fc01 = nn.Sequential(
            nn.Linear(7, 64),
            nn.LeakyReLU(0.2)
        )
        self.fc02 = nn.Sequential(
            nn.Linear(23, 128),
            nn.LeakyReLU(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, outputdim),
            nn.Sigmoid()
        )

        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.ReLU = nn.ReLU()

        outputdim0 = 64
        self.para_encoder = ParameterEncoderProcess(inputdim, 64, outputdim0)
        self.linear_merge = nn.Linear(outputdim0 * 3, outputdim0)
        self.linear_merge2 = nn.Linear(9, outputdim0)
        # self.linear_merge3 = nn.Sequential(nn.Linear(outputdim0, outputdim),nn.Sigmoid())

        # self.deconv0_1 = G_de_block(64, 64)  # 2
        self.deconv0_2 = G_de_block(16, 128)  # 4
        self.deconv0_3 = G_de_block(128, 256,3,1,1)  # 8
        self.deconv0_4 = G_de_block(256, 512,3,1,1)  # 8
        self.deconv0_5 = G_de_block(512, 1024,3,1,1)  # 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = G_block(64, 128)  #64
        self.conv3 = G_block(128, 256) #32
        self.conv4 = G_block(256, 256) #16
        self.conv5 = G_block(256, 512) #8
        self.conv6 = G_block(512, 512) #4

        self.deconv7 = G_de_block(512 + 1024, 512)  #8
        self.deconv8 = G_de_block(1024, 256)  #16
        self.deconv9 = G_de_block(256+256, 256)   #32
        self.deconv10 = G_de_block(256 + 256, 128)  #64
        self.deconv11 = G_de_block(128 + 128, 64)  #128
        self.deconv12 = G_de_block(64, 64)  # 256

        self.deconv13 =  nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

        # self.attn1 = SelfAttention(256 + 128, 8)


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x,wb,inimg):
        # img = self.fca(x)
        xin = x[:,7:]
        xout = x[:,:7]
        xmi = xout-xin
        feature_a = self.para_encoder(xin)
        feature_b = self.para_encoder(xout)
        feature_T = self.LReLU(self.linear_merge2(torch.cat([wb, xmi], dim=1)))
        concatenated = torch.cat([feature_a, feature_b,feature_T], dim=1)
        x1 = self.LReLU(self.linear_merge(concatenated))

        x00 = self.fc00(xin)
        x01 = self.fc01(xout)
        x02 = self.fc02(torch.cat([x,xmi,wb], dim=1))
        x1_ = self.fc1(torch.cat([x00,x01,x02], dim=1))
        x2_ = self.fc2(x1_)
        x3_ = self.fc3(x2_)
        x0 = x3_
        # x00 = self.linear_merge3(x1)

        x1 = x1.view(len(x1), -1, 2, 2) # 4 4 4
        x1 = self.deconv0_2(x1)
        x1 = self.deconv0_3(x1)
        x1 = self.deconv0_4(x1)
        x1 = self.deconv0_5(x1)

        out1_ = self.conv1(inimg)
        out1 = self.LReLU(out1_)  # 64 128 128
        out2_ = self.conv2(out1)
        out2 = self.LReLU(out2_)  # 128 64 64
        out3_ = self.conv3(out2)
        out3 = self.LReLU(out3_)  # 256 32 32
        out4_ = self.conv4(out3)
        out4 = self.LReLU(out4_)  # 512 16 16
        out5_ = self.conv5(out4)
        out5 = self.LReLU(out5_)  # 512 8 8
        out6_ = self.conv6(out5)
        out6 = self.LReLU(out6_)  # 512 4 4

        # decode
        out6 = torch.cat([out6,x1],dim = 1)
        out7_ = self.deconv7(out6)
        out7 = torch.cat((out7_, out5_), dim=1)
        out7 = self.ReLU(out7)  # 1024 8 8
        out8_ = self.deconv8(out7)
        out8 = torch.cat((out8_, out4_), dim=1)
        out8 = self.ReLU(out8)  # 768 16 16

        out9_ = self.deconv9(out8)
        out9 = torch.cat((out9_, out3_), dim=1)
        out9 = self.ReLU(out9)  # 384 32 32

        out10_ = self.deconv10(out9)
        out10_ = torch.cat((out10_, out2_), dim=1)
        out10 = self.ReLU(out10_)  # 192 64 64

        out11_ = self.deconv11(out10)
        out11 = self.ReLU(out11_)  # 64 128 128
        # out11 = self.attn2(out11)
        # out11 = torch.cat((out11, out1), dim=1)
        # out12 = self.deconv12(out11)
        out12_ = self.deconv12(out11)
        out12 = self.ReLU(out12_)
        out13 = self.deconv13(out12)

        return out13, x0