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


class SIBRNet(nn.Module):
    def __init__(self, config):
        super(SIBRNet, self).__init__()
        self.config = config
        self.n_views = self.config.num_input
        self.dcnet_mode = self.config.DCnet
        self.dcnet_ft = self.config.DCnet_ft


        self.enc_mode = self.config.Encoder
        self.dec_mode = self.config.Decoder

        self.num_res_blocks = 16
        self.nf = 64
        self.feature_extractor = SFE(num_res_blocks=self.num_res_blocks, n_feats=self.nf, res_scale=1)

        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(128)

        self.dcnet = self.get_dcnet()

        self.filter_layer = get_gaussian_kernel(kernel_size=101, channels=1)

        self.pcd_align_front = PCD_Align_IP(nf=self.nf, groups=8)

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

        # total_params = sum(p.numel() for p in self.rgb_alpha_tail.parameters())
        # print(f'{total_params:,} self.rgb_alpha_tail  total parameters.')
        # total_trainable_params = sum(
        #     p.numel() for p in self.rgb_alpha_tail.parameters() if p.requires_grad)
        # print(f'{total_trainable_params:,} self.rgb_alpha_tail training parameters.')

    def get_dcnet(self):
        if self.dcnet_mode == 'dcnet':
            from models.dcnet import DCnet
        else:
            raise Exception("invalid DCnet")

        dc_net = DCnet(self.config)
        self.dcnet_pretrained = os.path.join(HOME_PATH, 'output', self.dcnet_mode, 'model', f'checkpoint-{self.config.DCnet_restore_epoch}.ckpt')

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

    def get_ED_net(self, enc_mode, enc_in_channels, enc_out_channels):
        if enc_mode == 'None':
            enc_net = Identity()
        elif enc_mode == 'VGGUnet':
            enc_net = VGGUnet(net='vgg16', n_encoder_stages=3)
        elif enc_mode == 'unet':
            enc_net = UNet(
                in_channels=enc_in_channels,
                enc_channels=[64, 128, 256, 512],
                dec_channels=[256, 128, 64],
                out_channels=enc_out_channels,
                n_enc_convs=3,
                n_dec_convs=3,
            )
        else:
            raise Exception("invalid enc_net")

        return enc_net

    def change_ft_state(self, ft_flow):
        self.ft_flow = ft_flow
        if ft_flow == True:
            for p in self.flow_net.parameters():
                p.requires_grad = True
            self.flow_net.train()
        else:
            for p in self.flow_net.parameters():
                p.requires_grad = False
            self.flow_net.eval()

    def joint_normalize(self, I1, I2):
        shape = list(I1.size())
        I1_reshape = I1.reshape(shape[0], shape[1], -1)
        I2_reshape = I2.reshape(shape[0], shape[1], -1)
        I = torch.cat([I1_reshape, I2_reshape], dim=-1)
        m = I.mean(dim=-1, keepdim=True).unsqueeze(dim=-1)
        s = I.std(dim=-1, keepdim=True).unsqueeze(dim=-1)
        I1_norm = (I1-m)/(s+1e-6)
        I2_norm = (I2-m)/(s+1e-6)
        return I1_norm, I2_norm, m, s

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
        # src_gt_dms = data['src_gt_depths']
        # sparse_depth_masks = data['sparse_depth_masks']
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
        # confidence = depth_output['pred_out']
        confidence = depth_output['conf']

        # dense_pred_depth = dense_pred_depth.view(bs, nv, *dense_pred_depth.shape[1:])
        # confidence = confidence.view(bs, nv, *confidence.shape[1:])

        
        srcs = srcs.view(bs * nv, *srcs.shape[2:])
        srcs_fs = self.feature_extractor(srcs)
        srcs_fs = srcs_fs.view(bs, nv, *srcs_fs.shape[1:])

        dense_pred_depth = dense_pred_depth.view(bs * nv, *dense_pred_depth.shape[2:])
        dense_pred_depth = self.filter_layer(dense_pred_depth)
        dense_pred_depth = dense_pred_depth.view(bs, nv, *dense_pred_depth.shape[1:])

        src_wp_fs, wp_masks, src_pred_flow = getImage_forward(srcs_fs, dense_pred_depth.squeeze(
                        dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        sparse_src_wp_fs, sparse_wp_masks, sparse_src_pred_flow = getImage_forward(srcs_fs, src_dms.squeeze(
                dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        src_wp_pcd_fs = self.pcd_align_front(src_wp_fs, srcs_fs, sparse_wp_masks)
        
        src_wp_fs = sparse_src_wp_fs * sparse_wp_masks + src_wp_pcd_fs * (1 - sparse_wp_masks)
        # src_wp_fs = src_wp_fs.view(bs, nv * 3, h, w)
        # src_sparse_masks = sparse_wp_masks.view(bs, nv * 1, h, w)
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


