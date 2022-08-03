"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    KITTI Depth Completion Dataset Helper
"""


import os
import re
import cv2
import numpy as np
import json
import glob
import random
from numpy.core.defchararray import count
from numpy.core.fromnumeric import shape
from prefetch_generator import BackgroundGenerator
from torch.nn.modules import sparse
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF
import config


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def dataset_rng(idx):
    rng = np.random.RandomState()
    rng.seed(idx)
    return rng


class BaseDataset(Dataset):
    def __init__(self, name, train=True, logging_rate=16):
        self.name = name
        self.current_epoch = 0
        self.train = train
        self.logging_rate = logging_rate

    def base_len(self):
        raise NotImplementedError("")

    def base_getitem(self, idx, rng):
        raise NotImplementedError("")

    def __len__(self):
        return self.base_len()

    def __getitem__(self, idx):
        rng = dataset_rng(idx)
        idx = idx % len(self)
        return self.base_getitem(idx, rng)

class MultiDataset(Dataset):
    def __init__(self, name, *datasets, uniform_sampling=False):
        self.name = name
        self.datasets = []
        self.n_samples = []
        self.cum_n_samples = [0]
        self.uniform_sampling = uniform_sampling

        for dataset in datasets:
            self.append(dataset)

    @property
    def logging_rate(self):
        return min([dset.logging_rate for dset in self.datasets])

    @logging_rate.setter
    def logging_rate(self, logging_rate):
        for dset in self.datasets:
            dset.logging_rate = logging_rate

    def append(self, dataset):
        if not isinstance(dataset, BaseDataset):
            raise Exception("invalid Dataset in append")
        self.datasets.append(dataset)
        self.n_samples.append(len(dataset))
        n_samples = self.cum_n_samples[-1] + len(dataset)
        self.cum_n_samples.append(n_samples)

    def __len__(self):
        return self.cum_n_samples[-1]

    def __getitem__(self, idx):
        rng = dataset_rng(idx)
        if self.uniform_sampling:
            didx = rng.randint(0, len(self.datasets))
            sidx = rng.randint(0, self.n_samples[didx])
        else:
            idx = idx % len(self)
            didx = np.searchsorted(self.cum_n_samples, idx, side="right") - 1
            sidx = idx - self.cum_n_samples[didx]
        return self.datasets[didx].base_getitem(sidx, rng)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

meshGrids = {}

class FVS_own_Sparse(Dataset):
    def __init__(self,
                 root_path,
                 scale=0.5,
                 sparse='4',
                 patch_height=320,
                 patch_width=320,
                 padding=32,
                 n_nbs=4,
                 nbs_mode='near',
                 dilate_mask=False,
                 eval_seq='all',
                ):
        super(FVS_own_Sparse).__init__()

        self.root_path = root_path
        self.scale = scale
        self.sparse = sparse
        self.height = patch_height
        self.width = patch_width
        self.padding = padding
        self.n_nbs = n_nbs
        self.nbs_mode = nbs_mode
        self.dilate_mask = dilate_mask
        self.eval_seq = eval_seq
        
    # def get_val_dataset(self):
    #     self._val_Tanks_path = os.path.join(self.root_path, 'Val')
    #     print('Val dataset {} load…………'.format(self._val_Tanks_path))
    #     dsets = MultiDataset(name='val')
    #     for dset in os.listdir(self._val_Tanks_path):
    #         path = os.path.join(self._val_Tanks_path, dset, "s0.5")
    #         dsets.append(self.get_sub_set(dset, path, train=False, mode='all'))
    #     return dsets
    
    def get_test_dataset(self):
        self._test_path = os.path.join(self.root_path, 'Test')
        print('Test dataset {} load…………'.format(self._test_path))
        dsets = []
        for dset in config.FVS_own_scenes:
            path = os.path.join(self._test_path, dset)
            dsets.append(self.get_sub_set(dset, path, train=False))
        return dsets
    
    def get_sub_set(self, dset, scene_path, train=True, mode='all'):

        if self.sparse == '4':
            sparse_dm_paths = sorted(glob.glob(os.path.join(scene_path, 'depths', 'sparse_4', f"im_*.pfm")))
        elif self.sparse == '8':
            sparse_dm_paths = sorted(glob.glob(os.path.join(scene_path, 'depths', 'sparse_8', f"im_*.pfm")))
        elif self.sparse == 'all':
            sparse_dm_paths = sorted(glob.glob(os.path.join(scene_path, 'depths', 'dense', f"im_*.pfm")))
        else:
            raise Exception('invalid sparse mode')

        # im_paths = sorted(glob.glob(os.path.join(scene_path, 'images', f"im_*.jpg")))
        # dense_dm_paths = sorted(glob.glob(os.path.join(scene_path, 'depths', 'dense',f"im_*.pfm")))
        im_paths = sorted(glob.glob(os.path.join(scene_path, 'images', f"im_*.png")))
        im_paths += sorted(glob.glob(os.path.join(scene_path, 'images', f"im_*.jpg")))
        im_paths += sorted(glob.glob(os.path.join(scene_path, 'images', f"im_*.jpeg")))
        dense_dm_paths = sorted(glob.glob(os.path.join(scene_path, 'depths', 'dense', f"im_*.pfm")))
        Ks = np.load(os.path.join(scene_path, 'poses', 'dense', 'Ks.npy'))
        Rs = np.load(os.path.join(scene_path, 'poses', 'dense', 'Rs.npy'))
        ts = np.load(os.path.join(scene_path, 'poses', 'dense', 'ts.npy'))

        assert len(sparse_dm_paths) == len(dense_dm_paths)

        # print(f'{dset} all-sequence ...')
        tgt_ind = list(range(len(im_paths)))
        src_ind = list(range(len(im_paths)))

        counts = []
        # counts = counts[:, src_ind]

        dset = TanksSparse_subset(
            name=f'tat_{dset.replace("/", "_")}',
            tgt_im_paths=[im_paths[idx] for idx in tgt_ind],
            tgt_sparse_dm_paths=[sparse_dm_paths[idx] for idx in tgt_ind],
            tgt_dense_dm_paths=[dense_dm_paths[idx] for idx in tgt_ind],
            tgt_Ks=Ks[tgt_ind],
            tgt_Rs=Rs[tgt_ind],
            tgt_ts=ts[tgt_ind],
            tgt_counts=counts,
            src_im_paths=[im_paths[idx] for idx in src_ind],
            src_sparse_dm_paths=[sparse_dm_paths[idx] for idx in src_ind],
            src_dense_dm_paths=[dense_dm_paths[idx] for idx in src_ind],
            src_Ks=Ks[src_ind],
            src_Rs=Rs[src_ind],
            src_ts=ts[src_ind],
            patch_height=self.height,
            patch_width=self.width,
            padding=self.padding,
            n_nbs=self.n_nbs,
            nbs_mode=self.nbs_mode,
            dilate_mask=self.dilate_mask,
            train=train,
        )
        return dset
    
class TanksSparse_subset(BaseDataset):
    def __init__(self, 
                 name,
                 tgt_im_paths,
                 tgt_sparse_dm_paths,
                 tgt_dense_dm_paths,
                 tgt_Ks,
                 tgt_Rs,
                 tgt_ts,
                 tgt_counts,
                 src_im_paths,
                 src_sparse_dm_paths,
                 src_dense_dm_paths,
                 src_Ks,
                 src_Rs,
                 src_ts,
                 patch_height,
                 patch_width,
                 padding,
                 n_nbs,
                 nbs_mode,
                 dilate_mask,
                 train):
        super().__init__(name=name)

        self.tgt_im_paths = tgt_im_paths
        self.tgt_sparse_dm_paths = tgt_sparse_dm_paths
        self.tgt_dense_dm_paths = tgt_dense_dm_paths
        self.tgt_Ks = tgt_Ks
        self.tgt_Rs = tgt_Rs
        self.tgt_ts = tgt_ts
        self.tgt_counts = tgt_counts
        self.src_im_paths = src_im_paths
        self.src_sparse_dm_paths = src_sparse_dm_paths
        self.src_dense_dm_paths = src_dense_dm_paths
        self.src_Ks = src_Ks
        self.src_Rs = src_Rs
        self.src_ts = src_ts
        # self.patch = True
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.padding = padding
        self.n_nbs = n_nbs
        self.nbs_mode = nbs_mode
        self.dilate_mask = dilate_mask
        self.train = train
    
    def _load_data(self, p):
        _, ext = os.path.basename(p).split('.')
        if ext == 'pfm':
            data, _ = read_pfm(p)
            return data
        elif ext in ["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]:
            im = Image.open(p)
            im = im.convert('RGB')
            im = np.array(im)
            im = im.astype(np.float32) / 255
            im = im.transpose(2, 0, 1)
            return im
        elif ext == 'npy':
            data = np.load(p)
            return data
        else:
            raise Exception("invalid suffix")

    def pad(self, im):
        if self.padding is not None:
            h, w = im.shape[-2:]
            mh = h % self.padding
            ph = 0 if mh == 0 else self.padding - mh
            mw = w % self.padding
            pw = 0 if mw == 0 else self.padding - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im
    
    def load_pad(self, p):
        im = self._load_data(p)
        return self.pad(im)

    def base_len(self):
        return len(self.tgt_im_paths)

    def getTrans(self, mat_1, mat_2):
        mat_tmp = np.identity(4)
        mat_tmp[:3, :] = mat_1
        mat_1 = mat_tmp
        mat_tmp = np.identity(4)
        mat_tmp[:3, :] = mat_2
        mat_2 = mat_tmp
        del mat_tmp

        pose = np.reshape(np.matmul(mat_1, np.linalg.inv(mat_2)), [
            4, 4]).astype(np.float32)
        return pose[:3, :]

    def getPatchPixelCoords(self, h, w):
        if str((h, w)) not in meshGrids:
            h_grid = np.linspace(
                0, h-1, h, endpoint=True).reshape((h, 1, 1)).repeat(w, axis=1)
            w_grid = np.linspace(
                0, w-1, w, endpoint=True).reshape((1, w, 1)).repeat(h, axis=0)
            meshGrid = np.concatenate([w_grid, h_grid], axis=-1)
            meshGrids[str((h, w))] = meshGrid

        return meshGrids[str((h, w))]

    def base_getitem(self, idx, rng):
        # count = self.tgt_counts[idx]
        if self.nbs_mode == "argmax":            
            nbs = np.argsort(count)[::-1]
            nbs = nbs[: self.n_nbs]
            # nbs = list(nbs)
        elif self.nbs_mode == 'near':
            length = len(self.tgt_im_paths) - 1
            nbs_start = self.n_nbs // 2
            nbs_end = self.n_nbs + 1 - nbs_start
            nbs = [abs(i)+2 * min(length- i, 0)
                for i in range(idx-nbs_start, idx+nbs_end, 1)]
            nbs.remove(idx)
        else:
            raise Exception("invalid nbs_mode")

        ret = {}

        tgt_img_path = self.tgt_im_paths[idx]

        tgt_rgb = self.load_pad(self.tgt_im_paths[idx])
        tgt_K = self.tgt_Ks[idx]
        tgt_R = self.tgt_Rs[idx]
        tgt_t = self.tgt_ts[idx]

        h, w = tgt_rgb.shape[1:]

        src_rgbs = np.array([self.load_pad(self.src_im_paths[ii]) for ii in nbs])
        src_sparse_depths = np.array([cv2.resize(self.load_pad(self.src_sparse_dm_paths[ii]), (w, h)) for ii in nbs])
        # src_sparse_depths = np.resize(src_sparse_depths, (h, w))
        src_gt_depths = np.array([self.load_pad(self.src_dense_dm_paths[ii]) for ii in nbs])
        assert src_sparse_depths.shape == src_gt_depths.shape
        
        src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        src_ts = np.array([self.src_ts[ii] for ii in nbs])

        if self.train:
            # crop patch
            patch_h_from = rng.randint(0, h - self.patch_height + 1)
            patch_w_from = rng.randint(0, w - self.patch_width + 1)
            patch_h_to = patch_h_from + self.patch_height
            patch_w_to = patch_w_from + self.patch_width
            patch = np.array(
                (patch_h_from, patch_h_to, patch_w_from, patch_w_to),
                dtype=np.int32,
            )
        else:
            patch = np.array(
                (0, h, 0, w), dtype=np.int32
            )

        tgt_rgb = tgt_rgb[..., patch[0]:patch[1], patch[2]:patch[3]]

        src_rgbs = src_rgbs[..., patch[0]:patch[1], patch[2]:patch[3]]
        src_sparse_depths = src_sparse_depths[..., patch[0]:patch[1], patch[2]:patch[3]]
        src_gt_depths = src_gt_depths[..., patch[0]:patch[1], patch[2]:patch[3]]
        src_sparse_depths = src_sparse_depths[:, np.newaxis, ...]
        src_gt_depths = src_gt_depths[:, np.newaxis, ...]

        '''
        Note: we only use the sparse mask from sparse depth cause the depth value should be consistent with camera pose
        '''
        # sparse_depth_mask = np.ones_like(src_sparse_depths)
        # sparse_depth_mask[src_sparse_depths<0.0001] = 0
        d_valid = 0.0001
        sparse_depth_masks = src_sparse_depths > d_valid

        if self.train and self.dilate_mask:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            erode_masks = []
            for i in range(self.n_nbs):
                
                mask = cv2.erode(sparse_depth_masks[i].transpose(1, 2, 0).astype(np.float32), kernel)
                erode_masks.append(mask[np.newaxis, ...])
            sparse_depth_masks = np.stack(erode_masks, axis=0).astype(bool)

        src_sparse_depths = src_gt_depths * sparse_depth_masks


        # depth flow: src to tgt
        tgt_extrinsic = np.concatenate([tgt_R, tgt_t.reshape((3, 1))], axis=1)
        pose_trans_matrixs_src2tgt = []
        
        for i in range(self.n_nbs):
            src_R = src_Rs[i]
            src_t = src_ts[i]
            src_extrinsic = np.concatenate(
                [src_R, src_t.reshape((3, 1))], axis=1)
            matrix_src2tgt = self.getTrans(tgt_extrinsic, src_extrinsic)
            pose_trans_matrixs_src2tgt.append(matrix_src2tgt)
        pose_trans_matrixs_src2tgt = np.array(pose_trans_matrixs_src2tgt)

        pixel_coords = self.getPatchPixelCoords(h, w).astype(np.float32)
        patch_pixel_coords = pixel_coords[patch[0]:patch[1], patch[2]:patch[3], :]

        ret = {
                'tgt_rgb': tgt_rgb,
                'tgt_img_path': tgt_img_path,
                'src_rgbs': src_rgbs,
                'src_sparse_depths': src_sparse_depths,
                'sparse_depth_masks': sparse_depth_masks, 
                'src_gt_depths': src_gt_depths, 
                'tgt_K': tgt_K.astype(np.float32),
                'src_Ks': src_Ks.astype(np.float32),
                'pose_trans_matrixs_src2tgt': pose_trans_matrixs_src2tgt,
                'patch_pixel_coords': patch_pixel_coords,
            }

        return ret
