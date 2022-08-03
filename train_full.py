import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamax import Adamax
from visdom import Visdom
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import config
from config import argparser
from utils.warping import getImage_forward
from models.metric import *
from models.losses import *
from dataloader.tanks_sparse_dc import TanksSparse
from dataloader.tanks_sparse_dc import DataLoaderX

from models.model import SIBRNet

cmap = plt.get_cmap('plasma')

class VisdomWriter:
    def __init__(self, visdom_port):
        self.viz = Visdom(port=visdom_port)
        self.names = []

    def add_scalar(self, name, val, step):
        val = val.item()
        if name not in self.names:
            self.names.append(name)
            self.viz.line([val], [step], win=name, opts=dict(title=name))
        else:
            self.viz.line([val], [step], win=name, update='append')

    def add_image(self, name, image, step):
        self.viz.image(image, win=name, opts=dict(title=name))

    def close(self):
        return


class Trainer:
    def __init__(self,
                 config,
                 model,
                 dataset,
                 dataset_test):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.save_dir = self.config.save_dir
        self.max_epoch = self.config.max_epoch
        self.resume_epoch = self.config.resume_epoch
        self.batch_size = self.config.batch_size
        self.learing_rate = self.config.learning_rate
        self.lr_steps = self.config.lr_steps
        self.lr_gamma = self.config.lr_gamma
        self.visdom_port = self.config.visdom_port
        self.device = self.config.train_device
        self.prefix = self.config.prefix

        self.metric = {}

        model_checkpoint_path = os.path.join(
            self.save_dir, self.prefix, self.config.model_checkpoint)
        optimizer_checkpoint_path = os.path.join(
            self.save_dir, self.prefix, self.config.optimizer_checkpoint)
        if not os.path.exists(model_checkpoint_path):
            os.makedirs(model_checkpoint_path)
        if not os.path.exists(optimizer_checkpoint_path):
            os.makedirs(optimizer_checkpoint_path)
        self.model_checkpoint_path = model_checkpoint_path
        self.optimizer_checkpoint_path = optimizer_checkpoint_path

    def reload_checkpoint(self, root_path, epoch, model, optimizer, scheduler):
        print('reloading model epoch :{} …………'.format(epoch))
        model_checkpoint = torch.load(os.path.join(
            self.model_checkpoint_path, 'checkpoint-%d.ckpt' % epoch))
        train_checkpoint = torch.load(os.path.join(
            self.optimizer_checkpoint_path, 'checkpoint-%d.ckpt' % epoch))
        model.module.load_state_dict(model_checkpoint['state_dict'])
        optimizer_dict = train_checkpoint['optimizer']
        optimizer_dict['param_groups'][0]['lr'] = self.learing_rate
        optimizer.load_state_dict(optimizer_dict)
        scheduler.step(model_checkpoint['epoch'])
        assert model_checkpoint['step'] == train_checkpoint['step']
        return model_checkpoint['epoch'] + 1, model_checkpoint['step']

    def calc_loss(self, loss_fn, pred, gt, alpha):
        B = pred.size()[0]
        loss = loss_fn(pred, gt) * alpha
        return loss

    def image_l1_loss(self, pred, data, alpha):
        tgt = data['tgt_rgb']
        loss = torch.mean(torch.abs(tgt - pred)) * alpha
        return loss

    def image_perceptual_loss(self, loss_fn, pred, data, alpha):
        tgt = data['tgt_rgb']
        loss = loss_fn(pred, tgt) * alpha
        return loss

    def depth_l1_loss(self, pred, data, alpha):
        sparse_depth_mask = data['sparse_depth_masks'].clone()
        src_gt_depths = data['src_gt_depths'].clone()

        d_valid = 0.0001

        src_gt_mask = src_gt_depths > d_valid
        valid_mask = src_gt_mask

        pred = pred[valid_mask]
        gt = src_gt_depths[valid_mask]
        gt_tmp = gt.clone()
        gt_tmp[gt_tmp <= d_valid] = 1
        # loss = loss_fn(pred, gt)

        loss = torch.sum(torch.abs(pred - gt) / gt_tmp) / \
            (torch.sum(valid_mask) + 1e-2)

        return loss * alpha

    def depth_edge_loss(self, loss_fn, pred, data, alpha):
        sparse_depth_mask = data['sparse_depth_masks'].clone()
        src_gt_depths = data['src_gt_depths'].clone()
        bs, nv, _, h, w = src_gt_depths.shape

        d_valid = 0.0001

        src_gt_mask = src_gt_depths > d_valid
        valid_mask = src_gt_mask

        pred = pred * valid_mask
        gt = src_gt_depths * valid_mask
        # gt_tmp = gt.clone()
        # gt_tmp[gt_tmp <= d_valid] = 1
        # loss = loss_fn(pred, gt)
        gt = gt.view(bs * nv, *gt.shape[2:])
        pred = pred.view(bs * nv, *src_gt_depths.shape[2:])

        loss = loss_fn(gt, pred)

        return loss * alpha

        
        
    def depth_image_loss(self, depth, data, alpha=1):
        tgt_rgb = data['tgt_rgb'].clone()
        src_rgbs = data['src_rgbs'].clone()
        bs, nv, _, h, w = src_rgbs.shape

        gt_depth = data['src_gt_depths'].clone()
        sparse_depth = data['src_sparse_depths'].clone()

        tgt_K = data['tgt_K']
        src_Ks = data['src_Ks']
        pose_trans_matrixs_src2tgt = data['pose_trans_matrixs_src2tgt']
        patch_pixel_coords = data['patch_pixel_coords']

        src_wp_rgbs, wp_masks, src_pred_flow = getImage_forward(src_rgbs, depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        _, gt_masks, _ = getImage_forward(src_rgbs, gt_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        src_sparse_rgbs, sparse_masks, _ = getImage_forward(src_rgbs, sparse_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)
        
        final_src_sparse_rgbs = src_sparse_rgbs * sparse_masks + src_wp_rgbs * (1 - sparse_masks)

        gt_masks = gt_masks.view(bs * nv, *gt_masks.shape[2:])
        tgt_rgb = tgt_rgb.unsqueeze(dim=1)
        tgt_rgbs = tgt_rgb.expand(bs, nv, *tgt_rgb.shape[2:])
        tgt_rgbs = tgt_rgbs.contiguous().view(bs * nv, *tgt_rgbs.shape[2:])

        tgt_rgbs = tgt_rgbs * gt_masks
        final_src_sparse_rgbs = final_src_sparse_rgbs.view(
            bs * nv, *final_src_sparse_rgbs.shape[2:]) * gt_masks

        mse_loss = torch.nn.MSELoss(reduce=True, size_average=False)

        loss = mse_loss(final_src_sparse_rgbs, tgt_rgbs) / \
            (torch.sum(gt_masks) + 1e-2) * alpha
        # loss = mse_loss(src_wp_rgbs, tgt_rgbs) * alpha

        return loss, final_src_sparse_rgbs

    def im_to2np(self, im, mode='img'):
        if mode == 'img':
            im = im.detach().to("cpu").numpy()
            im = im.clip(0, 1)
            im = im.transpose(0, 2, 3, 1)
        elif mode == 'depth' or mode == 'conf':
            bs, nv, _, h, w = im.shape
            im = im.view(bs * nv, *im.shape[2:]).detach().to("cpu").numpy()
            im = im.transpose(0, 2, 3, 1)
        return im



    def depth_to_color(self, depth, d_min=None, d_max=None):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

    def train(self):
        self.model.train()
        # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nn.DataParallel(self.model, device_ids=[0]).cuda()
        # for p in self.model.module.dcnet.parameters():
        #     print(p.device)
        optimizer = Adamax(filter(lambda p: p.requires_grad, self.model.parameters()),
                           weight_decay=0, lr=self.learing_rate, betas=(0.9, 0.999))
        # optimizer = Adamax(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learing_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_steps, gamma=self.lr_gamma)

        writer = VisdomWriter(self.visdom_port)

        dataloader = DataLoaderX(self.dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=1, drop_last=True)

        step_one_epoch = len(self.dataset) // self.batch_size

        current_epoch = 0
        step = 0

        perceptual_loss = VGGPerceptualLoss()
        
        self.metric['rgb'] = MultipleMetric(
            metrics=[PSNRMetric(), SSIMMetric()])
        
        self.metric['dep'] = MultipleMetric(
            metrics=[RMSE(), MAE(), iRMSE(), iMAE()])

        if self.resume_epoch:
            current_epoch, step = self.reload_checkpoint(
                self.save_dir, self.resume_epoch, self.model, optimizer, scheduler)

        for epoch in range(current_epoch, self.max_epoch):
            total_loss = 0
            print('epoch: {}, current learning rate: {}'.format(
                epoch, scheduler.get_lr()[0]))
            progress = tqdm.tqdm(
                desc='training', total=step_one_epoch, ncols=75)
            for data in dataloader:
                step += 1

                for key, value in data.items():
                    if key != 'tgt_img_path':
                        data[key] = value.cuda()

                optimizer.zero_grad()

                output_dict = self.model(data)
                pred = output_dict['out']
                dense_pred_depth = output_dict['dense_pred_depth']
                confidence = output_dict['confidence']

                src_rgbs = data['src_rgbs']
                tgt_rgb = data['tgt_rgb']
                src_gt_depths = data['src_gt_depths']
                src_sparse_depths = data['src_sparse_depths']
                src_depths_masks = data['sparse_depth_masks']
                
                # depth_l1_loss = self.depth_l1_loss(dense_pred_depth, data, alpha=1)
                # depth_edge_loss = self.depth_edge_loss(edge_loss_fn, dense_pred_depth, data, alpha=1)
                _, src_wp_rgbs = self.depth_image_loss(dense_pred_depth, data, alpha=1)
                image_l1_loss = self.image_l1_loss(pred, data, alpha=1e2)
                image_perceptual_loss = self.image_perceptual_loss(perceptual_loss, pred, data, alpha=1e1)
                loss_total = image_l1_loss + image_perceptual_loss

                loss_total.backward()
                optimizer.step()
                # step_loss = loss_dict['loss'].cpu().detach().numpy()
                total_loss += loss_total.cpu().detach().numpy()

                # # metric
                # pred = self.im_to2np(pred, mode='img')
                # # dense_pred_depth = self.im_to2np(dense_pred_depth, mode='depth')
                # # confidence = self.im_to2np(confidence, mode='conf')
                # tgt_rgb = self.im_to2np(tgt_rgb, mode='img')
                # # src_gt_depths = self.im_to2np(src_gt_depths, mode='depth')
                
                # self.metric['rgb'].add(pred, tgt_rgb)

                # bs, nv, _, h, w = dense_pred_depth.shape
                # dense_pred_depth_tmp = dense_pred_depth.view(bs * nv, *dense_pred_depth.shape[2:]).detach().cpu()
                # src_gt_depths_tmp = src_gt_depths.view(bs * nv, *src_gt_depths.shape[2:]).detach().cpu()
                # self.metric['dep'].add(dense_pred_depth_tmp, src_gt_depths_tmp)

                rgb = tgt_rgb[0].detach().cpu()
                rgb_pred = pred[0].detach().cpu()
                src_wp_rgb = src_wp_rgbs[0].detach().cpu()
                # tgt = tgt_rgb[0].detach().cpu()
                sparse_depth = src_sparse_depths[0, 0, 0].detach().cpu().numpy()
                sparse_mask = src_depths_masks[0, 0, 0].detach().cpu().numpy()
                gt_depth = src_gt_depths[0, 0, 0].detach().cpu().numpy()
                depth_pred = dense_pred_depth[0, 0, 0].detach().cpu().numpy()
                confidence = confidence[0, 0, 0].detach().cpu().numpy()

                rgb = np.clip(rgb, a_min=0, a_max=1.0)
                rgb_pred = np.clip(rgb_pred, a_min=0, a_max=1.0)
                src_wp_rgb = np.clip(src_wp_rgb, a_min=0, a_max=1.0)
                sparse_depth = np.clip(
                    sparse_depth, a_min=0, a_max=self.config.max_depth)
                gt_depth = np.clip(gt_depth, a_min=0,
                                   a_max=self.config.max_depth)
                depth_pred = np.clip(depth_pred, a_min=0,
                                     a_max=self.config.max_depth)
                confidence = np.clip(confidence, a_min=0, a_max=1.0)

                sparse_depth = 255.0 * sparse_depth / self.config.max_depth
                gt_depth = 255.0 * gt_depth / self.config.max_depth
                depth_pred = 255.0 * depth_pred / self.config.max_depth
                confidence = 255.0 * confidence
                sparse_mask = 255.0 * sparse_mask

                sparse_depth = self.depth_to_color(sparse_depth.astype(
                    'uint8')).transpose(2, 0, 1)
                gt_depth = self.depth_to_color(gt_depth.astype('uint8')).transpose(2, 0, 1)
                depth_pred = self.depth_to_color(depth_pred.astype('uint8')).transpose(2, 0, 1)
                confidence = self.depth_to_color(confidence.astype('uint8')).transpose(2, 0, 1)
                # sparse_mask = sparse_mask.transpose(2, 0, 1)

                if step % 10 == 0:
                    # print(f'total loss={loss_total.item()}={depth_l1_loss.item()}+{depth_image_loss.item()}:' )
                    writer.add_scalar('loss/loss_total', loss_total, step)
                    writer.add_scalar('loss/image_l1_loss',
                                      image_l1_loss, step)
                    writer.add_scalar('loss/image_perceptual_loss',
                                      image_perceptual_loss, step)
                    writer.add_image('img/rgb', rgb.clamp(0, 1), step)
                    writer.add_image('img/rgb_pred', rgb_pred.clamp(0, 1), step)
                    writer.add_image('img/src_wp_rgb', src_wp_rgb.clamp(0, 1), step)
                    writer.add_image('img/sparse_depth', sparse_depth, step)
                    writer.add_image('img/gt_depth', gt_depth, step)
                    writer.add_image('img/depth_pred', depth_pred, step)
                    writer.add_image('img/confidence', confidence, step)
                    writer.add_image('img/sparse_mask', sparse_mask, step)

                progress.update(1)

            # saving checkpoint
            if epoch % 1 == 0:
                model_checkpoint = {
                    'state_dict': self.model.module.state_dict(),
                    'epoch': epoch,
                    'step': step
                }
                torch.save(model_checkpoint, os.path.join(
                    self.model_checkpoint_path, 'checkpoint-%d.ckpt') % epoch)
                train_checkpoint = {
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                }
                torch.save(train_checkpoint, os.path.join(
                    self.optimizer_checkpoint_path, 'checkpoint-%d.ckpt') % epoch)

            progress.close()
            print("Epoch: {}, loss: {}".format(
                epoch, total_loss / step_one_epoch))
            scheduler.step()
        writer.close()

def main():
    args = argparser(is_train=True)

    dataset_train = None
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
            step=args.step,
            dilate_mask=True,
            eval_seq=args.eval_seq,
        )
        dataset_train = dataset.get_train_dataset()
        dataset_test = dataset.get_test_dataset()
    else:
        raise Exception('Wrong Dataset')

    model = SIBRNet(args)
    print(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    trainer = Trainer(args, model, dataset_train, dataset_test)
    trainer.train()


if __name__ == '__main__':
    main()
