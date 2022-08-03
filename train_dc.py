import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader.tanks_sparse_dc import DataLoaderX
from torch.optim.adamax import Adamax
from visdom import Visdom
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import config
from config import argparser

from utils.warping import getImage_forward
from utils.util import flow_to_png, get_gaussian_kernel
from dataloader.tanks_sparse_dc import TanksSparse

from models.dcnet import DCnet
from models.losses import smooth_loss

# cm = plt.get_cmap('plasma')
cmap = plt.cm.jet

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

    def depth_l1_loss(self, pred, data, alpha):
        src_gt_depths = data['src_gt_depths'].clone()
        bs, nv, _, h, w = src_gt_depths.shape
        pred = pred.view(bs * nv, *pred.shape[2:])
        src_gt_depths = src_gt_depths.view(bs * nv, *src_gt_depths.shape[2:])

        d_valid = 0.0001

        src_gt_mask = src_gt_depths > d_valid
        valid_mask = src_gt_mask

        pred = pred[valid_mask]
        gt = src_gt_depths[valid_mask]
        gt_tmp = gt.clone()
        gt_tmp[gt_tmp <= d_valid] = d_valid
        # loss = loss_fn(pred, gt)

        loss = torch.sum(torch.abs(pred - gt) / gt_tmp) / (torch.sum(valid_mask) + 1)

        return loss * alpha
    
    def depth_input_l2_loss(self, pred, data, alpha):
        src_gt_depths = data['src_gt_depths'].clone()
        sparse_depth_mask = data['sparse_depth_masks'].clone()
        bs, nv, c, h, w = src_gt_depths.shape
        pred = pred.view(bs * nv, c, h, w)
        src_gt_depths = src_gt_depths.view(bs * nv, c, h, w)
        
        d_valid = 0.0001
        src_gt_mask = src_gt_depths > d_valid
        valid_mask = src_gt_mask

        pred = pred[valid_mask]
        src_gt_depths = src_gt_depths[valid_mask]

        gt_tmp = src_gt_depths.clone()
        gt_tmp[gt_tmp <= d_valid] = d_valid

        loss_fn = torch.nn.MSELoss(reduction='none')

        loss = torch.sum(loss_fn(pred, src_gt_depths) / gt_tmp) / (torch.sum(valid_mask) + 1)

        return loss * alpha

    def depth_image_loss(self, depth, data, alpha=1):
        tgt_rgb = data['tgt_rgb'].clone()
        src_rgbs = data['src_rgbs'].clone()
        bs, nv, _, h, w = src_rgbs.shape

        # depth = depth.view(bs, nv, *depth.shape)
        gt_depth = data['src_gt_depths'].clone()
        sparse_depth = data['src_sparse_depths'].clone()

        tgt_K = data['tgt_K']
        src_Ks = data['src_Ks']
        pose_trans_matrixs_src2tgt = data['pose_trans_matrixs_src2tgt']
        patch_pixel_coords = data['patch_pixel_coords']

        blur_layer = get_gaussian_kernel(kernel_size=101, channels=1).cuda()
        depth = depth.view(bs * nv, *depth.shape[2:])
        depth = blur_layer(depth)
        depth = depth.view(bs, nv, *depth.shape[1:])

        src_wp_rgbs, wp_masks, src_pred_flow = getImage_forward(src_rgbs, depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        _, gt_masks, _ = getImage_forward(src_rgbs, gt_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)
        
        src_sparse_rgbs, sparse_masks, _ = getImage_forward(src_rgbs, sparse_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)
        
        # final_src_sparse_rgbs = src_sparse_rgbs * sparse_masks + src_wp_rgbs * (1 - sparse_masks)
        final_src_sparse_rgbs = src_wp_rgbs
        # d_valid = 0.0001
        # src_gt_mask = depth > d_valid

        # left_img = src_rgbs[0, 0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # right_img = src_rgbs[0, 1].permute(1, 2, 0).detach().cpu().numpy() * 255.0

        # left_wp_img = src_wp_rgbs[0, 0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # right_wp_img = src_wp_rgbs[0, 1].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # left_wp_img = left_wp_img.clip(0, 255)
        # right_wp_img = right_wp_img.clip(0, 255)

        # left_wp_img_sparse = src_sparse_rgbs[0, 0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # right_wp_img_sparse = src_sparse_rgbs[0, 1].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # left_wp_img_sparse = left_wp_img_sparse.clip(0, 255)
        # right_wp_img_sparse = right_wp_img_sparse.clip(0, 255)

        # left_wp_img_final = final_src_sparse_rgbs[0, 0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # right_wp_img_final = final_src_sparse_rgbs[0, 1].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        # left_wp_img_final = left_wp_img_final.clip(0, 255)
        # right_wp_img_final = right_wp_img_final.clip(0, 255)


        # left_depth_mask = wp_masks[0, 0, 0].detach().cpu().numpy() * 255.0
        # right_depth_mask = wp_masks[0, 1, 0].detach().cpu().numpy() * 255.0

        # left_gt_mask = gt_masks[0, 1, 0].detach().cpu().numpy() * 255.0
        # right_gt_mask = gt_masks[0, 1, 0].detach().cpu().numpy() * 255.0

        # tgt = tgt_rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0

        # cv2.imwrite('left_img.jpg', left_img)
        # cv2.imwrite('right_img.jpg', right_img)
        # cv2.imwrite('left_wp_img.jpg', left_wp_img)
        # cv2.imwrite('right_wp_img.jpg', right_wp_img)
        # cv2.imwrite('left_wp_img_sparse.jpg', left_wp_img_sparse)
        # cv2.imwrite('right_wp_img_sparse.jpg', right_wp_img_sparse)
        # cv2.imwrite('left_wp_img_final.jpg', left_wp_img_final)
        # cv2.imwrite('right_wp_img_final.jpg', right_wp_img_final)
        # cv2.imwrite('left_depth_mask.jpg', left_depth_mask)
        # cv2.imwrite('right_depth_mask.jpg', right_depth_mask)
        # cv2.imwrite('left_gt_mask.jpg', left_gt_mask)
        # cv2.imwrite('right_gt_mask.jpg', right_gt_mask)
        # cv2.imwrite('tgt.jpg', tgt)
        # exit(0)

        # tenLinear = softsplat.FunctionSoftsplat(tenInput=src_rgbs, tenFlow=src_pred_flow, tenMetric=None, strType='linear')
        # gt_masks = gt_masks.view(bs * nv, *gt_masks.shape[2:])
        gt_masks= gt_masks.expand(bs, nv, 3, *gt_masks.shape[3:]).type(torch.BoolTensor)

        tgt_rgb = tgt_rgb.unsqueeze(dim=1)
        tgt_rgbs = tgt_rgb.expand(bs, nv, *tgt_rgb.shape[2:])
        # tgt_rgbs = tgt_rgbs.contiguous().view(bs * nv, *tgt_rgbs.shape[2:])

        tgt_rgbs_tmp = tgt_rgbs[gt_masks]
        final_src_sparse_rgbs_tmp = final_src_sparse_rgbs[gt_masks]
        # mse_loss = torch.nn.MSELoss(reduce=True, size_average=False)

        # print(mse_loss(final_src_sparse_rgbs_tmp, tgt_rgbs_tmp))
        # print((torch.sum(gt_masks) + 1))
        loss = torch.sum(torch.abs(final_src_sparse_rgbs_tmp - tgt_rgbs_tmp)) / (torch.sum(gt_masks) + 1)
        # loss = mse_loss(src_wp_rgbs, tgt_rgbs) * alpha

        return loss * alpha, final_src_sparse_rgbs

    def depth_flow_loss(self, depth, data, alpha=1):
        tgt_rgb = data['tgt_rgb'].clone()
        src_rgbs = data['src_rgbs'].clone()
        bs, nv, _, h, w = src_rgbs.shape

        # depth = depth.view(bs, nv, *depth.shape)
        gt_depth = data['src_gt_depths'].clone()
        sparse_depth = data['src_sparse_depths'].clone()

        tgt_K = data['tgt_K']
        src_Ks = data['src_Ks']
        pose_trans_matrixs_src2tgt = data['pose_trans_matrixs_src2tgt']
        patch_pixel_coords = data['patch_pixel_coords']

        _, _, src_pred_flow = getImage_forward(src_rgbs, depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)

        _, _, src_gt_flow = getImage_forward(src_rgbs, gt_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)
        
        _, _, sparse_flow = getImage_forward(src_rgbs, sparse_depth.squeeze(
            dim=2), tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=patch_pixel_coords)
        
        d_valid = 0.0001
        src_gt_masks = gt_depth > d_valid

        src_pred_flow = src_pred_flow.permute(0, 1, 4, 2, 3)
        src_gt_flow = src_gt_flow.permute(0, 1, 4, 2, 3)
        sparse_flow = sparse_flow.permute(0, 1, 4, 2, 3)


        src_gt_masks= src_gt_masks.expand(bs, nv, 2, *src_gt_masks.shape[3:]).cuda()
        pred_mask_flow = src_pred_flow * src_gt_masks
        gt_mask_flow = src_gt_flow * src_gt_masks
        sparse_mask_flow = sparse_flow * src_gt_masks

        pred_mask_flow_tmp = src_pred_flow[src_gt_masks]
        gt_mask_flow_tmp = src_gt_flow[src_gt_masks]
        sparse_mask_flow_tmp = sparse_flow[src_gt_masks]

        # pred_flow_mask = src_pred_flow[src_gt_masks]
        # gt_flow_mask = src_gt_flow[src_gt_masks]

        # loss_fn = torch.nn.l1
        # loss = loss_fn(pred_mask_flow, gt_mask_flow)
        loss = torch.sum(torch.abs(pred_mask_flow_tmp - gt_mask_flow_tmp)) / (torch.sum(src_gt_masks) + 1)

        # pred = pred[valid_mask]
        # gt = src_gt_depths[valid_mask]
        # gt_tmp = gt.clone()
        # gt_tmp[gt_tmp <= d_valid] = d_valid
        # # loss = loss_fn(pred, gt)

        # loss = torch.sum(torch.abs(pred - gt) / gt_tmp) / (torch.sum(valid_mask) + 1)

        return loss * alpha, gt_mask_flow, pred_mask_flow, sparse_mask_flow


    def depth_smooth_loss(self, depth, rgb, alpha):
        bs, nv, _, h, w = depth.shape

        depth = depth.view(bs * nv, 1, h, w)
        rgb = rgb.view(bs * nv, 3, h, w)

        loss = smooth_loss(depth, rgb)

        return loss * alpha

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
        optimizer = Adamax(filter(lambda p: p.requires_grad, self.model.parameters()),
                           weight_decay=0, lr=self.learing_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_steps, gamma=self.lr_gamma)

        writer = VisdomWriter(self.visdom_port)

        dataloader = DataLoaderX(self.dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=2, drop_last=True)

        step_one_epoch = len(self.dataset) // self.batch_size

        current_epoch = 0
        step = 0

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
                depth = output_dict['depth']
                esti_conf = output_dict['esti_conf']
                dc_conf = output_dict['dc_conf']
                dc_depth = output_dict['dc_depth']
                de_depth = output_dict['esti_depth']

                src_rgbs = data['src_rgbs']
                tgt_rgb = data['tgt_rgb']
                src_gt_depths = data['src_gt_depths']
                src_sparse_depths = data['src_sparse_depths']
                src_depths_masks = data['sparse_depth_masks']


                depth_input_l2_loss = self.depth_input_l2_loss(depth, data, alpha=10)
                dc_depth_input_l2_loss = self.depth_input_l2_loss(dc_depth, data, alpha=10)
                de_depth_input_l2_loss = self.depth_input_l2_loss(de_depth, data, alpha=10)

                depth_smooth_loss = self.depth_smooth_loss(depth, src_rgbs, alpha=40)
                
                depth_image_loss, src_wp_rgbs = self.depth_image_loss(depth, data, alpha=10)
                depth_flow_loss, src_gt_flow, src_pred_flow, sparse_flow = self.depth_flow_loss(depth, data, alpha=5e-3)

                loss_total = depth_input_l2_loss + depth_smooth_loss + dc_depth_input_l2_loss + de_depth_input_l2_loss
                # loss_total = depth_input_l2_loss + depth_smooth_loss

                loss_total.backward()
                optimizer.step()
                # step_loss = loss_dict['loss'].cpu().detach().numpy()
                total_loss += loss_total.cpu().detach().numpy()

                # rgb = src_rgbs[0, 0].detach().cpu()
                rgb = tgt_rgb[0].detach().cpu()
                src_wp_rgb = src_wp_rgbs[0, 0].detach().cpu()
                # src_sparse_rgb = src_sparse_rgbs[0].detach().cpu()
                sparse_depth = src_sparse_depths[0, 0, 0].detach().cpu().numpy()
                sparse_mask = src_depths_masks[0, 0, 0].detach().cpu().numpy()
                gt_depth = src_gt_depths[0, 0, 0].detach().cpu().numpy()
                depth_pred = depth[0, 0, 0].detach().cpu().numpy()
                esti_conf = esti_conf[0, 0, 0].detach().cpu().numpy()
                dc_conf = dc_conf[0, 0, 0].detach().cpu().numpy()
                dc_depth = dc_depth[0, 0, 0].detach().cpu().numpy()
                de_depth = de_depth[0, 0, 0].detach().cpu().numpy()

                pred_flow = src_pred_flow[0, 0].detach().cpu().numpy()
                gt_flow = src_gt_flow[0, 0].detach().cpu().numpy()
                sparse_flow = sparse_flow[0, 0].detach().cpu().numpy()
                pred_flow = flow_to_png(pred_flow).transpose(2, 0, 1)
                gt_flow = flow_to_png(gt_flow).transpose(2, 0, 1)
                sparse_flow = flow_to_png(sparse_flow).transpose(2, 0, 1)


                rgb = np.clip(rgb, a_min=0, a_max=1.0)
                src_wp_rgb = np.clip(src_wp_rgb, a_min=0, a_max=1.0)
                # src_sparse_rgb = np.clip(src_sparse_rgb, a_min=0, a_max=1.0)
                
                sparse_depth = np.clip(
                    sparse_depth, a_min=0, a_max=self.config.max_depth)
                gt_depth = np.clip(gt_depth, a_min=0,
                                   a_max=self.config.max_depth)
                depth_pred = np.clip(depth_pred, a_min=0,
                                     a_max=self.config.max_depth)
                dc_depth = np.clip(dc_depth, a_min=0,
                                     a_max=self.config.max_depth)
                de_depth = np.clip(de_depth, a_min=0,
                                     a_max=self.config.max_depth)

                esti_conf = np.clip(esti_conf, a_min=0, a_max=1.0)
                dc_conf = np.clip(dc_conf, a_min=0, a_max=1.0)

                sparse_depth = 255.0 * sparse_depth / self.config.max_depth
                gt_depth = 255.0 * gt_depth / self.config.max_depth
                depth_pred = 255.0 * depth_pred / self.config.max_depth
                dc_depth = 255.0 * dc_depth / self.config.max_depth
                de_depth = 255.0 * de_depth / self.config.max_depth
                esti_conf = 255.0 * esti_conf
                dc_conf = 255.0 * dc_conf
                sparse_mask = 255.0 * sparse_mask

                sparse_depth = self.depth_to_color(sparse_depth.astype(
                    'uint8')).transpose(2, 0, 1)
                gt_depth = self.depth_to_color(gt_depth.astype('uint8')).transpose(2, 0, 1)
                depth_pred = self.depth_to_color(depth_pred.astype('uint8')).transpose(2, 0, 1)
                dc_depth = self.depth_to_color(dc_depth.astype('uint8')).transpose(2, 0, 1)
                de_depth = self.depth_to_color(de_depth.astype('uint8')).transpose(2, 0, 1)

                # sparse_mask = sparse_mask.transpose(2, 0, 1)

                if step % 10 == 0:
                    # print(f'total loss={loss_total.item()}={depth_l1_loss.item()}+{depth_image_loss.item()}:' )
                    writer.add_scalar('loss/loss_total', loss_total, step)
                    writer.add_scalar('loss/depth_smooth_loss',
                                      depth_smooth_loss, step)
                    writer.add_scalar('loss/depth_input_l2_loss',
                                      depth_input_l2_loss, step)
                    writer.add_scalar('loss/dc_depth_input_l2_loss',
                                      dc_depth_input_l2_loss, step)
                    writer.add_scalar('loss/de_depth_input_l2_loss',
                                      de_depth_input_l2_loss, step)
                    writer.add_scalar('loss/depth_flow_loss',
                                      depth_flow_loss, step)
                    writer.add_image('img/rgb', rgb.clamp(0, 1), step)
                    writer.add_image('img/src_wp_rgb', src_wp_rgb.clamp(0, 1), step)
                    writer.add_image('img/pred_flow', pred_flow, step)
                    writer.add_image('img/gt_flow', gt_flow, step)
                    writer.add_image('img/sparse_flow', sparse_flow, step)
                    # writer.add_image('img/src_sparse_rgb', src_sparse_rgb.clamp(0, 1), step)
                    writer.add_image('img/sparse_depth', sparse_depth, step)
                    writer.add_image('img/gt_depth', gt_depth, step)
                    writer.add_image('img/depth_pred', depth_pred, step)
                    writer.add_image('img/dc_depth', dc_depth, step)
                    writer.add_image('img/de_depth', de_depth, step)
                    writer.add_image('img/esti_conf', esti_conf, step)
                    writer.add_image('img/dc_conf', dc_conf, step)
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
            dilate_mask=True,
            eval_seq=args.eval_seq,
        )
        dataset_train = dataset.get_train_dataset()
        dataset_test = dataset.get_test_dataset()
    else:
        raise Exception('Wrong Dataset')

    args.max_epoch = 60
    args.lr_steps = [50, 55]

    model = DCnet(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


    trainer = Trainer(args, model, dataset_train, dataset_test)
    trainer.train()


if __name__ == '__main__':
    main()
