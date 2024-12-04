import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import crnn
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=None, s=None, p=None):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=s, padding=p)
        self.BN = nn.GroupNorm(num_channels=out_channels, num_groups=1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
    def forward(self, input):
        out = self.relu(self.dropout(self.BN(self.conv(input))))
        return out

class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=None, s=None, p=None):
        super(DeConvLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=s, padding=p)
        self.BN = nn.GroupNorm(num_channels=out_channels, num_groups=1)
        self.dropout = nn.Dropout(p=0)
        self.relu = nn.ReLU()
    def forward(self, input):
        out = self.relu(self.dropout(self.BN(self.conv(input))))
        # out = self.conv(input)
        # out = self.BN(out)
        # out = self.dropout(out)
        # out = self.relu(out)
        return out

class encoder(nn.Module):
    def __init__(self, in_channels, time_len):
        super(encoder, self).__init__()
        self.time_len = time_len
        self.in_conv0_0 = ConvLayer(in_channels=in_channels, out_channels=64, k=1,s=1,p=0)
        self.in_conv0_1 = ConvLayer(in_channels=64, out_channels=64, k=1, s=1, p=0)
        self.in_conv1_0 = ConvLayer(in_channels=in_channels, out_channels=16, k=3, s=1, p=1)
        self.in_conv1_1 = ConvLayer(in_channels=16, out_channels=32, k=3, s=1, p=1)
        self.in_conv1_2 = ConvLayer(in_channels=32, out_channels=32, k=3, s=1, p=1)
        self.in_conv1_3 = ConvLayer(in_channels=32, out_channels=64, k=3, s=1, p=1)
        self.in_conv2 = ConvLayer(in_channels=64, out_channels=64, k=3, s=1, p=0)

    def forward(self, x):
        for i in range(self.time_len):
            t_resdual = self.in_conv0_1(self.in_conv0_0(x[i]))
            t = self.in_conv1_3(self.in_conv1_2(self.in_conv1_1(self.in_conv1_0(x[i]))))
            t1 = t_resdual + t
            t2 = self.in_conv2(t1)
            t3 = self.in_conv2(t2)

            t1 = torch.unsqueeze(t1, 0)
            t2 = torch.unsqueeze(t2, 0)
            t3 = torch.unsqueeze(t3, 0)
            if i == 0:
                hx1 = t1
                hx2 = t2
                hx3 = t3
            else:
                hx1 = torch.cat((hx1, t1), dim=0)
                hx2 = torch.cat((hx2, t2), dim=0)
                hx3 = torch.cat((hx3, t3), dim=0)
        return hx1, hx2, hx3

class decoder(nn.Module):
    def __init__(self, in_channels, time_len):
        super(decoder, self).__init__()
        self.time_len = time_len
        self.deconv3_2 = DeConvLayer(in_channels=64, out_channels=64, k=3, s=1, p=0)
        self.deconv2_1 = DeConvLayer(in_channels=64, out_channels=64, k=3, s=1, p=0)
        self.in_conv2 = ConvLayer(in_channels=128, out_channels=64, k=3, s=1, p=1)
        self.in_conv1 = ConvLayer(in_channels=128, out_channels=64, k=3, s=1, p=1)

    def forward(self, hx1, hx2, hx3):
        for i in range(self.time_len):
            x2 = self.deconv3_2(hx3[i])
            x2 = torch.cat((hx2[i], x2), dim=1)
            x2 = self.in_conv2(x2)

            x1 = self.deconv2_1(x2)
            x1 = torch.cat((hx1[i], x1), dim=1)
            x1 = self.in_conv1(x1)
            x1 = torch.unsqueeze(x1, 0)
            if i == 0:
                hx = x1
            else:
                hx = torch.cat((hx, x1), dim=0)
        return hx

def warp(x, flow):
    n, c, h, w = x.size()

    norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
    col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
    row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
    grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
    grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm
    output = F.grid_sample(x, grid, align_corners=True)
    return output

class FDAF(nn.Module):
    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='IN'), act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, stride=1, padding=(5 - 1) // 2, bias=True, groups=in_channels * 2),
            nn.GroupNorm(num_channels=in_channels*2, num_groups=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 16, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(16, 2, kernel_size=1, padding=0, bias=False))

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        x2_feat = warp(x2, flow) - x1
        return x2_feat

class ConvDisRNN(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvDisRNN, self).__init__()
        self.time_len = time_len
        self.batch_size = batch_size
        self.encoder = encoder(in_channels=in_channels, time_len=time_len)
        self.decoder = decoder(in_channels=64, time_len=time_len)
        self.neck_layer = FDAF(in_channels=64)
        self.fc = nn.Linear((self.time_len-1) * 64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hx1, hx2, hx3 = self.encoder(x)
        hx = self.decoder(hx1, hx2, hx3)
        flow = torch.zeros((self.time_len - 1, hx1.shape[1], hx1.shape[2], hx1.shape[3], hx1.shape[4])).cuda()
        for i in range(self.time_len - 1):
            flow[i] = self.neck_layer(hx1[i], hx1[i + 1])
        flow = flow[:,:,:,2,2].squeeze()
        flow = flow.permute(1, 0, 2).contiguous().view(self.batch_size, -1)
        flow = self.fc(flow)
        flow = self.sigmoid(flow)
        return flow