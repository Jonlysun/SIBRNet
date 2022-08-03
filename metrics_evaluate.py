import glob
import os
from PIL import Image
import cv2
import math
import numpy as np
import shutil as shutil
import argparse
import lpips
import config

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = np.mean((img1 - img2)**2)
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_lpips(loss_fn, img1_path, img2_path, use_gpu=True):
    # use_gpu = True
    img1_tensor = lpips.im2tensor(lpips.load_image(img1_path))
    img2_tensor = lpips.im2tensor(lpips.load_image(img2_path))
    # print(img1_tensor.shape)
    # print(img2_tensor.shape)
    # exit(0)
    if use_gpu:
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()
    dist = loss_fn.forward(img1_tensor, img2_tensor)
    loss = dist.mean().item()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['TanksSparse', 'FVS_own_Sparse', 'Surround'])
    parser.add_argument('--sparse', type=str, choices=['4', '8', '16', 'all'])
    parser.add_argument('--result_name', type=str)

    args = parser.parse_args()

    if args.dataset == 'TanksSparse':
        scenes = ['Train', 'Playground', 'M60', 'Truck']
        gt_path = os.path.join(config.Tanks_and_Temples_root, 'Test')
        save_root = './Result/TanksSparse'
    elif args.dataset == 'FVS_own_Sparse':
        scenes = ['bike', 'flowers', 'pirate', 'playground', 'sandbox', 'soccertable']
        gt_path = config.FVS_root
        save_root = './Result/FVSSparse'
    elif args.dataset == 'Surround':
        scenes = ['basketball', 'meetingroom', 'park', 'philosopher', 'soccer', 'statue']
        gt_path = config.Surround_root
        save_root = './Result/Surround'

    save_path = os.path.join(save_root, args.result_name)
    method = f'SIBRNet-sparse-{args.sparse}'

    total_str_rgb = ''
    total_str_y = ''

    file_txt=open(os.path.join(save_root, f'./{method}-result.txt'), mode='w')

    file_txt.write('#' * 80 + '\n')
    file_txt.write(save_path + '\n')

    total_count = 0.0
    total_psnr_rgb = 0.0
    total_ssim_rgb = 0.0
    total_psnr_y = 0.0
    total_ssim_y = 0.0
    total_lpips = 0.0


    str_rgb = method
    str_y = method

    use_gpu=True
    spatial = True
    loss_fn = lpips.LPIPS(net='alex', spatial=spatial)
    if use_gpu:
        loss_fn.cuda()

    for dataset_count, scene in enumerate(scenes):

        tmp_total_count = 0.0
        tmp_total_psnr_rgb = 0.0
        tmp_total_ssim_rgb = 0.0
        tmp_total_psnr_y = 0.0
        tmp_total_ssim_y = 0.0
        tmp_total_lpips = 0.0
    
        img_path = os.path.join(save_path, scene)
        
        if args.dataset == 'TanksSparse':
            gt_scene_path = os.path.join(gt_path, scene)
            gt_l = sorted(glob.glob(os.path.join(gt_scene_path, 's0.5', 'images', f'im_*.jpg')))

            image_l = sorted(glob.glob(os.path.join(img_path, f'im_*.jpg')))
        elif args.dataset == 'FVS_own_Sparse':
            gt_scene_path = os.path.join(gt_path, scene)
            gt_l = sorted(glob.glob(os.path.join(gt_scene_path, 'images', f'im_*.jpg')))

            image_l = sorted(glob.glob(os.path.join(img_path, f'im_*.jpg')))
        elif args.dataset == 'Surround':            
            gt_scene_path = os.path.join(gt_path, scene)
            gt_l = sorted(glob.glob(os.path.join(gt_scene_path, 'images',  f'*.jpg')))

            image_l = sorted(glob.glob(os.path.join(img_path, f'*.jpg')))

        
        assert len(image_l) == len(gt_l), f'pred number: {len(image_l)}, gt number: {len(gt_l)}'
        
        for i, gt_image_path in enumerate(gt_l):
            pred_image_path = image_l[i]
            gt_image = Image.open(gt_image_path)
            sr_image = Image.open(image_l[i])

            gt_image = np.array(gt_image)
            sr_image = np.array(sr_image)
            psnr_rgb = calculate_psnr(sr_image, gt_image)
            ssim_rgb = calculate_ssim(sr_image, gt_image)
            lpips_rgb = calculate_lpips(loss_fn, pred_image_path, gt_image_path, use_gpu=use_gpu)
            gt_image = bgr2ycbcr(gt_image, only_y=True)
            sr_image = bgr2ycbcr(sr_image, only_y=True)
            psnr_y = calculate_psnr(sr_image, gt_image)
            ssim_y = calculate_ssim(sr_image, gt_image)

            tmp_total_count = tmp_total_count + 1
            tmp_total_psnr_rgb = tmp_total_psnr_rgb + psnr_rgb
            tmp_total_ssim_rgb = tmp_total_ssim_rgb + ssim_rgb
            tmp_total_psnr_y = tmp_total_psnr_y + psnr_y
            tmp_total_ssim_y = tmp_total_ssim_y + ssim_y
            tmp_total_lpips = tmp_total_lpips + lpips_rgb

            print(image_l[i])
            # print('psnr_rgb:', psnr_rgb)
            # print('ssim_rgb:', ssim_rgb)
            # print('psnr_y:', psnr_y)
            # print('ssim_y:', ssim_y)

        total_count = total_count + tmp_total_count
        total_psnr_rgb = total_psnr_rgb + tmp_total_psnr_rgb
        total_ssim_rgb = total_ssim_rgb + tmp_total_ssim_rgb
        total_psnr_y = total_psnr_y + tmp_total_psnr_y
        total_ssim_y = total_ssim_y + tmp_total_ssim_y
        total_lpips = total_lpips + tmp_total_lpips

        print('*' * 80)
        print(scene)
        print('psnr_rgb:', tmp_total_psnr_rgb / tmp_total_count)
        print('ssim_rgb:', tmp_total_ssim_rgb / tmp_total_count)
        print('psnr_y:', tmp_total_psnr_y / tmp_total_count)
        print('ssim_y:', tmp_total_ssim_y / tmp_total_count)
        print('lpips:', tmp_total_lpips / tmp_total_count)

        str_rgb = str_rgb + ' & ' + str(round(tmp_total_psnr_rgb / tmp_total_count, 2))
        str_rgb = str_rgb + ' & ' + str(round(tmp_total_ssim_rgb / tmp_total_count, 4))
        str_rgb = str_rgb + ' & ' + str(round(tmp_total_lpips / tmp_total_count, 4))
        str_y = str_y + ' & ' + str(round(tmp_total_psnr_y / tmp_total_count, 2))
        str_y = str_y + ' & ' + str(round(tmp_total_ssim_y / tmp_total_count, 4))
        str_y = str_y + ' & ' + str(round(tmp_total_lpips / tmp_total_count, 4))

        file_txt.write('*' * 80 + '\n')
        file_txt.write(scene + '\n')
        file_txt.write('psnr_rgb :' + str(tmp_total_psnr_rgb / tmp_total_count) + '\n')
        file_txt.write('ssim_rgb :' + str(tmp_total_ssim_rgb / tmp_total_count) + '\n')
        file_txt.write('psnr_y :' + str(tmp_total_psnr_y / tmp_total_count) + '\n')
        file_txt.write('ssim_y :' + str(tmp_total_ssim_y / tmp_total_count) + '\n')
        file_txt.write('lpips :' + str(tmp_total_lpips / tmp_total_count) + '\n')

    str_rgb = str_rgb + ' & ' + str(round(total_psnr_rgb / total_count, 2))
    str_rgb = str_rgb + ' & ' + str(round(total_ssim_rgb / total_count, 4))
    str_rgb = str_rgb + ' & ' + str(round(total_lpips / total_count, 4))
    str_y = str_y + ' & ' + str(round(total_psnr_y / total_count, 2))
    str_y = str_y + ' & ' + str(round(total_ssim_y / total_count, 4))
    str_y = str_y + ' & ' + str(round(total_lpips / total_count, 4))

    str_rgb = str_rgb + ' \\\\ \n'
    str_y = str_y + ' \\\\ \n'

    total_str_rgb = total_str_rgb + str_rgb
    total_str_y = total_str_y + str_y

    print('*' * 80)
    print('total result')
    print('psnr_rgb:', total_psnr_rgb / total_count)
    print('ssim_rgb:', total_ssim_rgb / total_count)
    print('psnr_y:', total_psnr_y / total_count)
    print('ssim_y:', total_ssim_y / total_count)
    print('lpips:', total_lpips / total_count)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total result' + '\n')
    file_txt.write('psnr_rgb :' + str(total_psnr_rgb / total_count) + '\n')
    file_txt.write('ssim_rgb :' + str(total_ssim_rgb / total_count) + '\n')
    file_txt.write('psnr_y :' + str(total_psnr_y / total_count) + '\n')
    file_txt.write('ssim_y :' + str(total_ssim_y / total_count) + '\n')
    file_txt.write('lpips :' + str(total_lpips / total_count) + '\n')

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_rgb' + '\n')
    file_txt.write(total_str_rgb)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_y' + '\n')
    file_txt.write(total_str_y)

    file_txt.close()