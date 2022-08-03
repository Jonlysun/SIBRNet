import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random
from torch.autograd import Variable
# from pwcnet import PWCNet
import sys


model_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet34': 'pretrained/resnet34.pth'
}

def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=pretrained)
    return net

def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=pretrained)
    return net

def upsample2d_as(inputs, target_as, mode='bilinear'):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=1, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    # padding = (kernel + 1) * stride // 2
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=1, output_padding=1,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers
    
# def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bn=True):
#     if bn:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                       dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                       dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class ResBlockIP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockIP, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x, y):
        x1 = x
        out = self.conv1(torch.cat([x, y], dim=1))
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x1
        return out


backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat(
            [tenHorizontal, tenVertical], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return F.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='zeros')


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 enc_channels=[64, 128, 256],
                 dec_channels=[128, 64],
                 out_channels=3,
                 n_enc_convs=2,
                 n_dec_convs=2):
        super(UNet, self).__init__()

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

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x

class VGGUnet(nn.Module):
    def __init__(
        self, net='vgg16', pool='average', n_encoder_stages=3, n_decoder_convs=2
    ):
        super(VGGUnet, self).__init__()

        if net == 'vgg16':
            vgg = torchvision.models.vgg16(pretrained=True).features
        elif net == 'vgg19':
            vgg = torchvision.models.vgg19(pretrained=True).features
        else:
            raise Exception('invalid vgg net')

        encs = []
        enc = []
        encs_channels = []
        channels = -1
        for mod in vgg:
            if isinstance(mod, nn.Conv2d):
                channels = mod.out_channels

            if isinstance(mod, nn.MaxPool2d):
                encs.append(nn.Sequential(*enc))
                encs_channels.append(channels)
                n_encoder_stages -= 1
                if n_encoder_stages <= 0:
                    break
                if pool == "average":
                    enc = [
                        nn.AvgPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                elif pool == "max":
                    enc = [
                        nn.MaxPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                else:
                    raise Exception("invalid pool")
            else:
                enc.append(mod)
        self.encs = nn.ModuleList(encs)

        cin = encs_channels[-1] + encs_channels[-2]
        decs = []
        for idx, cout in enumerate(reversed(encs_channels[:-1])):
            decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
            cin = cout + encs_channels[max(-idx - 3, -len(encs_channels))]
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x):
        feats = []
        for enc in self.encs:
            x = enc(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = F.interpolate(
                x0, size=(x1.shape[2], x1.shape[3]), mode='nearest'
            )
            x = torch.cat((x0, x1), dim=1)
            x = dec(x)
            feats.append(x)

        x = feats.pop()
        return x

class ConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        nonlinearity="tanh",
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.conv_gates = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=2 * channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_can = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        else:
            raise Exception("invalid nonlinearity")

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                (x.shape[0], self.channels_out, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([x, h], dim=1)
        combined_conv = torch.sigmoid(self.conv_gates(combined))
        del combined
        r = combined_conv[:, : self.channels_out]
        z = combined_conv[:, self.channels_out :]

        combined = torch.cat([x, r * h], dim=1)
        n = self.nonlinearity(self.conv_can(combined))
        del combined

        h = z * h + (1 - z) * n
        return h

class GRUUNet(nn.Module):
    def __init__(
        self,
        channels_in,
        enc_channels=[32, 64, 64],
        dec_channels=[64, 32],
        n_enc_convs=2,
        n_dec_convs=2,
        gru_all=False,
        gru_nonlinearity="relu",
        bias=False,
    ):
        super().__init__()
        self.n_rnn = 0
        self.gru_nonlinearity = gru_nonlinearity

        stride = 1
        cin = channels_in
        encs = []
        for cout in enc_channels:
            encs.append(
                self._enc(
                    cin,
                    cout,
                    stride=stride,
                    n_convs=n_enc_convs,
                    gru_all=gru_all,
                )
            )
            stride = 2
            cin = cout
        self.encs = nn.ModuleList(encs)

        cin = enc_channels[-1] + enc_channels[-2]
        decs = []
        for idx, cout in enumerate(dec_channels):
            decs.append(
                self._dec(cin, cout, n_convs=n_dec_convs, gru_all=gru_all)
            )
            cin = cout + enc_channels[max(-idx - 3, -len(enc_channels))]
        self.decs = nn.ModuleList(decs)

    def _enc(
        self, channels_in, channels_out, stride=2, n_convs=2, gru_all=False
    ):
        mods = []
        if stride > 1:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(
                    ConvGRU2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        nonlinearity=self.gru_nonlinearity,
                    )
                )
            else:
                mods.append(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                mods.append(nn.ReLU())
            channels_in = channels_out
            stride = 1
        return nn.Sequential(*mods)

    def _dec(self, channels_in, channels_out, n_convs=2, gru_all=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(
                    ConvGRU2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        nonlinearity=self.gru_nonlinearity,
                    )
                )
            else:
                mods.append(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x, hs=None):
        if hs is None:
            hs = [None for _ in range(self.n_rnn)]

        hidx = 0
        feats = []
        for enc in self.encs:
            for mod in enc:
                if isinstance(mod, ConvGRU2d):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = F.interpolate(
                x0, size=(x1.shape[2], x1.shape[3]), mode="nearest"
            )
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, ConvGRU2d):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        x = feats.pop()
        return x, hs


