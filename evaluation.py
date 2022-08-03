import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import tqdm
import config
from config import argparser
import matplotlib.pyplot as plt
from PIL import Image
from dataloader.tanks_sparse_dc import TanksSparse
from dataloader.fvs_sparse_dc import FVS_own_Sparse
from dataloader.surround_sparse_dc import Surround

from models.model import SIBRNet

cmap = plt.cm.jet

def state_dict_reload(state_dict):
    new_state_dict = {}

    for key in state_dict.keys():
        if 'depth_esti_net' in key:
            new_key = key.replace('depth_esti_net', 'global_net')
            new_state_dict[new_key] = state_dict[key]
        elif 'depth_complet_net' in key:
            new_key = key.replace('depth_complet_net', 'local_net')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict

class Evaler():
    
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.metric_mode = config.metric
        self.batch_size = config.batch_size
        self.save_dir = config.save_dir
        self.prefix = config.prefix
        self.model_checkpoint = config.model_checkpoint
        self.optimizer_checkpoint = config.optimizer_checkpoint
        self.restore_epoch = config.restore_epoch
        self.visualize = config.visualize
        self.eval_seq = config.eval_seq
    
        self.checkpoint_path = os.path.join(self.save_dir, self.prefix, self.model_checkpoint, 'final-model.ckpt')
        print(f'{self.checkpoint_path} is restoring ......')
        self.checkpoint = torch.load(self.checkpoint_path)

        self.model.load_state_dict(self.checkpoint['state_dict'])

        self.device = torch.device('cuda:0')

    def im_tonp(self, im):
        im = im.permute(0, 2, 3, 1)
        im = im.detach().cpu().numpy()
        return im

    def depth_to_color(self, depth, d_min=None, d_max=None):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

    def eval_run(self):

        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.config.dataset == 'TanksSparse':
            root_path = os.path.join('./Result/TanksSparse', f'{self.prefix}_sparse_{self.config.sparse}_input_{self.config.num_input}_s{self.config.scale}_result')
            TEST_SCENE = config.Tank_scenes
        elif self.config.dataset == 'FVS_own_Sparse':
            root_path = os.path.join('./Result/FVSSparse', f'{self.prefix}_sparse_{self.config.sparse}_input_{self.config.num_input}_s{self.config.scale}_result')
            TEST_SCENE = config.FVS_own_scenes
        elif self.config.dataset == 'Surround':
            root_path = os.path.join('./Result/Surround', f'{self.prefix}_sparse_{self.config.sparse}_input_{self.config.num_input}_s{self.config.scale}_result')
            # TEST_SCENE = os.listdir(os.path.join(config.FVS_own_sparse_root, 'Test'))
            TEST_SCENE = config.Surround_scenes
        
        errs_list = []
        with torch.no_grad():
            for eval_set_idx, eval_set in enumerate(self.dataset):
                eval_dataloader = DataLoader(eval_set, batch_size=self.batch_size, shuffle=False, 
                                            num_workers=2, drop_last=False, pin_memory=True)
                step_one_epoch = len(eval_set) // self.batch_size

                print(f'{TEST_SCENE[eval_set_idx]} is evaluating ......')

                errs_list = []
                progress = tqdm.tqdm(desc='Evaluating', total=step_one_epoch, ncols=75)

                save_dir = os.path.join(root_path, TEST_SCENE[eval_set_idx])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)


                # Test_visual = 'Visual_Test'
                for batch_idx, data in enumerate(eval_dataloader):

                    # visual_path = os.path.join(Test_visual, str(batch_idx))
                    # os.makedirs(visual_path, exist_ok=True)

                    for key in data.keys():
                        if key != 'tgt_img_path':
                            data[key] = data[key].to(self.device)    

                    img_paths = data['tgt_img_path']
                    img = Image.open(img_paths[0])
                    H, W, _ = np.array(img).shape
                    del img

                    output_dict = self.model(data)
                    pred = output_dict['out']

                    tgt = data['tgt_rgb']

                    pred = pred[..., :H, :W]
                    tgt = tgt[..., :H, :W]

                    pred = self.im_tonp(pred)
                    tgt = self.im_tonp(tgt)

                    pred = np.clip(pred, 0, 1)
                    pred = pred * 255.0
                    bz = pred.shape[0]
                    for i in range(bz):
                        img_path = img_paths[i].split('/')[-1]
                        img_path = os.path.join(save_dir, img_path)
                        Image.fromarray(pred[i].astype(np.uint8)).save(img_path)


                    progress.update(1)
                progress.close()

def main():
    args = argparser(is_train=True)
    print(args)

    dataset_test = None
    if args.dataset == 'TanksSparse':
        dataset = TanksSparse(
            root_path=config.Tanks_and_Temples_root,
            scale=args.scale,
            sparse=args.sparse,
            patch_height=args.patch_height,
            patch_width=args.patch_width,     
            padding=32,
            n_nbs=args.num_input,
            nbs_mode=args.tanks_train_nbs_mode,
            dilate_mask=True,
            eval_seq=args.eval_seq,
        )
        dataset_test = dataset.get_test_dataset()

    elif args.dataset == 'FVS_own_Sparse':
        dataset = FVS_own_Sparse(
            root_path=config.FVS_root,
            scale=args.scale,
            sparse=args.sparse,
            patch_height=args.patch_height,
            patch_width=args.patch_width,     
            padding=32,
            n_nbs=args.num_input,
            nbs_mode='near',
        )
        dataset_test = dataset.get_test_dataset()

    elif args.dataset == 'Surround':
        dataset = Surround(
            root_path=config.Surround_root,
            scale=args.scale,
            sparse=args.sparse,
            patch_height=args.patch_height,
            patch_width=args.patch_width,     
            padding=32,
            n_nbs=args.num_input,
            nbs_mode='near',
        )
        dataset_test = dataset.get_test_dataset()

    else:
        raise Exception('Wrong Dataset')

    model = SIBRNet(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    evaler = Evaler(args, model, dataset_test)
    evaler.eval_run()

if __name__ == '__main__':
    main()