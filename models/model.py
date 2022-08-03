from tokenize import group
from turtle import forward
from sklearn.semi_supervised import SelfTrainingClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_utils import *
from utils.warping import getImage_forward
# from convlstm import ConvBLSTM
# from convlstm import BiDeformableConvLSTM, BiGuidedConvLSTM
from convlstm import ConvBLSTM, PCD_Align_IP
from config import *
from utils.util import get_gaussian_kernel


import os 
DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_SCALE = 100


def state_dict_reload(state_dict):
    new_state_dict = {}

    for key in state_dict.keys():
        if 'depth_esti_net' in key:
            new_key = key.replace('depth_esti_net', 'global_net')
            new_state_dict[new_key] = state_dict[key]
        elif 'depth_complet_net' in key:
            new_key = key.replace('depth_complet_net', 'local_net')
            new_state_dict[new_key] = state_dict[key]

    return new_state_dict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args


class Tail(nn.Module):
    def __init__(self, in_channel, out_channel, block_num=4):
        super().__init__()

        self.tail = nn.ModuleList()
        for i in range(block_num):
                self.tail.append(ResBlock(in_channels=in_channel, out_channels=in_channel))
        
        self.tail.append(conv3x3(in_channel, out_channel))

    def forward(self, x):
        for module in self.tail:
            x = module(x)
        return x

class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

class BiasCorrect(nn.Module):
    def __init__(self, nf, groups):
        super(BiasCorrect, self).__init__()

        self.nf = nf
        self.groups = groups
        self.pcd_align_front = PCD_Align_IP(nf=self.nf, groups=8)

    def forward(self, sparse_src_wp_fs, src_wp_fs, srcs_fs, sparse_wp_masks):
        src_wp_pcd_fs = self.pcd_align_front(src_wp_fs, srcs_fs, sparse_wp_masks)
        
        src_wp_fs = sparse_src_wp_fs * sparse_wp_masks + src_wp_pcd_fs * (1 - sparse_wp_masks)

        return src_wp_fs

class SIBRNet(nn.Module):
    def __init__(self, config):
        super(SIBRNet, self).__init__()
        self.config = config
        self.n_views = self.config.num_input
        self.dcnet_mode = self.config.DCnet
        self.dcnet_ft = self.config.DCnet_ft
        self.num_res_blocks = 16
        self.nf = 64
        self.feature_extractor = SFE(num_res_blocks=self.num_res_blocks, n_feats=self.nf, res_scale=1)

        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(128)

        self.dcnet = self.get_dcnet()

        self.filter_layer = get_gaussian_kernel(kernel_size=101, channels=1)

        self.bcm = BiasCorrect(nf=self.nf, groups=8)

        self.patch_size = (self.config.patch_height, self.config.patch_width)
        self.groups = 8

        self.ConvBLSTM = ConvBLSTM(
            input_size=self.patch_size,
            input_dim=65,
            hidden_dim=hidden_dim,
            kernel_size=(3,3),
            num_layers=1,
            batch_first=True,
            # return_all_layers=True,
        )

        self.rgb_conv = nn.Sequential(
            conv3x3(256, 128),
            nn.ReLU(inplace=True),
            conv3x3(128, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, 3)
            )
        
        self.alpha_conv = nn.Sequential(
            conv3x3(256, 128),
            nn.ReLU(inplace=True),
            conv3x3(128, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, 1)
            )
        self.print_parameters()


        # self.avgpool = nn.AdaptiveAvgPool2d()

    def print_parameters(self):

        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f'{total_params:,} self.feature_extractor  total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.feature_extractor training parameters.')

        total_params = sum(p.numel() for p in self.dcnet.parameters())
        print(f'{total_params:,} self.dcnet  total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.dcnet.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.dcnet training parameters.')

        total_params = sum(p.numel() for p in self.ConvBLSTM.parameters())
        print(f'{total_params:,} self.ConvBLSTM  total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.ConvBLSTM.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.ConvBLSTM training parameters.')

    def get_dcnet(self):
        if self.dcnet_mode == 'dcnet':
            from models.dcnet import DCnet
        else:
            raise Exception("invalid DCnet")

        dc_net = DCnet(self.config)
        self.dcnet_pretrained = os.path.join(HOME_PATH, 'output', self.dcnet_mode, 'model', 'dcnet-final.ckpt')

        if self.dcnet_pretrained is not None:
            assert os.path.exists(self.dcnet_pretrained), "file not found: {}".format(self.dcnet_pretrained)
            print(f'{self.dcnet_pretrained} is loading......')

            checkpoint = torch.load(self.dcnet_pretrained)
            dc_net.load_state_dict(checkpoint['state_dict'])

            print('Load network parameters from : {}'.format(self.dcnet_pretrained))

            
        if not self.dcnet_ft:
            for p in dc_net.parameters():
                p.requires_grad = False
            dc_net.eval()
        else:
            for p in dc_net.parameters():
                p.requires_grad = True
                print(p.device)
                    # dc_net.cuda(0)


        return dc_net

    def forward(self, data):
        """
        srcs: [B, n_views, C, H, W]
        src_dms: [B, n_views, 1, H, W]
        src_Ks: [B, n_views, 3, 3]
        tgt:  [B, C, H, W]
        tgt_K: [B, 3, 3]
        patch_pixel_coords: [B, H, W, 2]
        pose_trans: [B, n_views, 3, 4]
        """
        srcs = data['src_rgbs']
        src_dms = data['src_sparse_depths']
        src_Ks = data['src_Ks']
        tgt_K = data['tgt_K']
        patch_pixel_coords = data['patch_pixel_coords']
        pose_trans_matrixs_src2tgt = data['pose_trans_matrixs_src2tgt']

        bs, nv, _, h, w = srcs.shape

        if not self.dcnet_ft:
            with torch.no_grad():
                depth_output = self.dcnet(data)
        else:

            depth_output = self.dcnet(data)
        dense_pred_depth = depth_output['depth']
        confidence = depth_output['conf']
    
        dense_pred_depth = dense_pred_depth.view(bs * nv, *dense_pred_depth.shape[2:])
        dense_pred_depth = self.filter_layer(dense_pred_depth)
        dense_pred_depth = dense_pred_depth.view(bs, nv, *dense_pred_depth.shape[1:])

        srcs = srcs.view(bs * nv, *srcs.shape[2:])
        srcs_fs = self.feature_extractor(srcs)
        srcs_fs = srcs_fs.view(bs, nv, *srcs_fs.shape[1:])

        src_wp_fs, _, src_pred_flow = getImage_forward(srcs_fs, dense_pred_depth.squeeze(
                        dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        sparse_src_wp_fs, sparse_wp_masks, _ = getImage_forward(srcs_fs, src_dms.squeeze(
                dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        src_wp_fs = self.bcm(sparse_src_wp_fs, src_wp_fs, srcs_fs, sparse_wp_masks)

        src_wp_fs = torch.cat([src_wp_fs, sparse_wp_masks], dim=2)
        
        reversed_idx = list(reversed(range(src_wp_fs.shape[1])))
        src_wp_fs_rev = src_wp_fs[:, reversed_idx, ...]
        feats = self.ConvBLSTM(src_wp_fs, src_wp_fs_rev)

        # Aggregate
        rgbs = []
        alphas = []
        for i in range(nv):
            feature = feats[:, i]
            rgbs.append(self.rgb_conv(feature))
            alphas.append(self.alpha_conv(feature))

        rgbs = torch.stack(rgbs)
        alphas = torch.stack(alphas)
        alphas = torch.softmax(alphas, dim=0)
        x = (alphas * rgbs).sum(dim=0)
        del rgbs, alphas
        
        output_dict = {
            'out': x,
            'dense_pred_depth': dense_pred_depth,
            'src_pred_flow': src_pred_flow,
            'confidence': confidence,
        }

        return output_dict


