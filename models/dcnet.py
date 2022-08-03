"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
# from .ERFNet import Net
from modules_utils import *

class GlobalNet(nn.Module):
    def __init__(self,
                 in_channels,
                 enc_channels=[64, 128, 256],
                 dec_channels=[128, 64],
                 out_channels=3,
                 n_enc_convs=2,
                 n_dec_convs=2):
        super(GlobalNet, self).__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x):
        outs = []
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        feats = []
        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
            feats.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        
        feat1, feat2, feat3 = feats[0], feats[1], feats[2]
        return x, feat1, feat2, feat3

class LocalNet(nn.Module):
    def __init__(self, in_channels=2, num_res=4):
        super().__init__()
        nf = 32
        self.init_layer = conv_bn_relu(in_channels, nf, 3, bn=False)

        self.layer1 = nn.ModuleList()
        for i in range(num_res):
            self.layer1.append(ResBlock(nf * 2, nf * 2))
        self.layer1_tail = conv_bn_relu(nf * 2, nf * 2, 3, stride=2, bn=False)

        self.layer2 = nn.ModuleList()
        for i in range(num_res):
            self.layer2.append(ResBlock(nf * 4, nf * 4))
        self.layer2_tail = conv_bn_relu(nf * 4, nf * 4, 3, stride=2, bn=False)

        self.layer3 = convt_bn_relu(nf * 8, nf * 4, 3, stride=2, bn=False)        
        self.layer4 = convt_bn_relu(nf * 4, nf, 3, stride=2, bn=False)

        self.fuse = conv_bn_relu(nf, 2, 3, bn=False, relu=False)

    def forward(self, x, feat1, feat2, feat3):
        out0 = self.init_layer(x)

        out = torch.cat([out0, feat3], dim=1)
        for m in self.layer1:
            out = m(out)
        out = self.layer1_tail(out)

        out = torch.cat([out, feat2], dim=1)
        for m in self.layer2:
            out = m(out)
        out = self.layer2_tail(out)

        out = torch.cat([out, feat1], dim=1)
        out = self.layer3(out)
        out = self.layer4(out)

        out1 = out0 + out

        out1 = self.fuse(out1)
        return out1

class DCnet(nn.Module):
    def __init__(self, args):
        super(DCnet, self).__init__()
        out_chan = 2

        self.args = args

        combine = 'concat'
        in_channels = 5
        out_channels = 1
        thres = 15
        num_res = 4
        self.combine = combine
        self.in_channels = in_channels
        self.out_channels = out_channels

        out_channels = 3
        # self.depthnet = Net(in_channels=in_channels, out_channels=out_channels)
        self.global_net = GlobalNet(in_channels=5, 
                                           enc_channels=[32, 64, 128, 256], 
                                           dec_channels=[128, 64, 32],
                                           out_channels=2, 
                                           n_enc_convs=2,
                                           n_dec_convs=2)

        self.local_net = LocalNet(in_channels=2, num_res=num_res)
                    
        self.thres = thres
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data, epoch=50):
        # if self.in_channels > 1:
        #     rgb_in = input[:, 1:, :, :]
        #     lidar_in = input[:, 0:1, :, :]
        # else:
        #     lidar_in = input

        rgb_in = data['src_rgbs']
        depth = data['src_sparse_depths']
        mask = data['sparse_depth_masks']

        bs, nv, _, h, w = rgb_in.shape
        rgb_in = rgb_in.view(bs * nv, *rgb_in.shape[2:])
        depth = depth.view(bs * nv, *depth.shape[2:])
        mask = mask.view(bs * nv, *mask.shape[2:])

        input = torch.cat([rgb_in, depth, mask], dim=1)

        # 1. Depth Estimation NET
        out, feat1, feat2, feat3= self.global_net(input)
        esti_depth = out[:, 0:1, :, :]
        esti_conf = out[:, 1:, :, :]

        input = torch.cat((depth, mask), 1)

        # 3. Depth Completion NET
        out = self.local_net(input, feat1, feat2, feat3)
        dc_depth = out[:, 0:1, :, :]
        dc_conf = out[:, 1:, :, :]

        # 4. Late Fusion
        esti_conf, dc_conf = torch.chunk(self.softmax(torch.cat((esti_conf, dc_conf), 1)), 2, dim=1)
        out = esti_conf * esti_depth + dc_conf * dc_depth

        depth = out.view(bs, nv, *out.shape[1:])
        esti_conf = esti_conf.view(bs, nv, *esti_conf.shape[1:])
        dc_conf = dc_conf.view(bs, nv, *dc_conf.shape[1:])
        esti_depth = esti_depth.view(bs, nv, *esti_depth.shape[1:])
        dc_depth = dc_depth.view(bs, nv, *dc_depth.shape[1:])

        output = {
            'depth': depth,
            'conf': esti_conf,
            'esti_conf': esti_conf,
            'dc_conf': dc_conf,
            'esti_depth': esti_depth,
            'dc_depth': dc_depth,
        }

        return output




if __name__ == '__main__':
    batch_size = 4
    in_channels = 4
    H, W = 256, 1216
    model = DCnet(in_channels).cuda()
    print(model)
    print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in model.parameters())/1e6))
    input = torch.rand((batch_size, in_channels, H, W)).cuda().float()
    out = model(input)
    print(out[0].shape)
