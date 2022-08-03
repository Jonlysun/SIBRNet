import re
import torch
import math
import torch.nn as nn
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2
import math
from PIL import Image, ImageDraw
from datetime import datetime
import shutil
from skimage.restoration import denoise_bilateral

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def img_HWC2CHW(x): return x.permute(2, 0, 1)
def gray2rgb(x): return x.unsqueeze(2).repeat(1, 1, 3)


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)
def mse2psnr(x): return -10. * np.log(x+TINY_NUMBER) / np.log(10.)

def poly_mask(H, W, size=5):
    y_coor = np.random.randint(0, H-1, size=size)
    x_coor = np.random.randint(0, W-1, size=size)
    cor_xy = np.dstack((x_coor, y_coor)).astype(np.int32)
    im = np.zeros((H, W), dtype='uint8')
    cv2.polylines(im, cor_xy, 1, 1)
    cv2.fillPoly(im, cor_xy, 1)
    mask_array = 1 - im
    return mask_array

def poly_erosion_mask(H, W, size=5, kernel_size=3):
    '''
    BLACKHAT operation
    '''
    # mask_poly = poly_mask(H, W, size=5)

    mask = poly_mask(H, W, size=size)
    mask = 1 - mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))                  #定义矩形结构元素
    # TOPHAT_img = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)     #顶帽运算
    # BLACKHAT_mask = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel) #黒帽运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    mask = 1 - mask

    return mask

def brush_stroke_mask(H, W, min_num_vertex=4, max_num_vertex=12, mean_angle=2*math.pi / 5, angle_range=2*math.pi / 15, min_width=12, max_width=40):
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(
                    2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)),
                       int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    # mask = np.reshape(mask, (1, H, W, 1))
    return mask




def generate_depthmap(depth_prob, min_disp, disp_step, depht_at_inifinity,
                      color_max_val=None, use_argmax=True):
    if use_argmax:
        depth_idx = np.argmax(depth_prob, axis=0)
    else:
        depth_idx = np.argmin(depth_prob, axis=0)

    img_depth = depth_idx*disp_step
    zero_disp = (depth_idx == 0)
    img_depth = 1.0 / (img_depth + min_disp)
    img_depth[zero_disp] = depht_at_inifinity

    if color_max_val is None:
        img_depth_colored, color_max_val = apply_colormap_to_depth(
            img_depth, depht_at_inifinity)
        return img_depth, img_depth_colored, zero_disp, color_max_val

    else:
        img_depth_colored = apply_colormap_to_depth(
            img_depth, depht_at_inifinity, max_depth=color_max_val)
        return img_depth, img_depth_colored, zero_disp


def apply_colormap_to_depth(img_depth, depth_at_infinity, max_depth=None, max_percent=95, RGB=True):
    img_depth_colored = img_depth.copy()
    m = np.min(img_depth_colored)
    M = np.max(img_depth_colored)

    if max_depth is None:
        valid_mask = img_depth_colored < depth_at_infinity
        valid_mask = np.logical_and(
            valid_mask, np.logical_not(np.isinf(img_depth)))
        valid_mask = np.logical_and(valid_mask, img_depth != 0.0)
        list_data = img_depth[valid_mask]

        hist, bins = np.histogram(list_data, bins=20)
        n_data = len(list_data)
        threshold_max = n_data * float(max_percent)/100.0
        sum_hist = 0

        for bin_idx, hist_val in enumerate(hist):
            sum_hist += hist_val
            if sum_hist > threshold_max:
                M = bins[bin_idx + 1]
                break
    else:
        M = max_depth

    img_depth_colored[img_depth_colored > M] = M
    img_depth_colored = (img_depth_colored - m) / (M - m)
    img_depth_colored = (img_depth_colored * 255).astype(np.uint8)
    img_depth_colored = cv2.applyColorMap(img_depth_colored, cv2.COLORMAP_HSV)

    if RGB:
        img_depth_colored = cv2.cvtColor(img_depth_colored, cv2.COLOR_BGR2RGB)

    if max_depth is None:
        return img_depth_colored, M
    else:
        return img_depth_colored

 
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


def save_pfm(filename, image, scale=1):

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(
        image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def flow_to_png(flow_map, max_value=None):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new

# tensor


def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel,
                         iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)


def depthFilter(disp):
    # Filter the depth
    noisy = disp
    result = denoise_bilateral(
        noisy, sigma_color=0.5, sigma_spatial=4, win_size=7, multichannel=False)
    b = np.percentile(noisy, list(range(100)))
    a = np.percentile(result, list(range(100)))
    x = estimate_coef(a, b)
    result = (result * x[1] + x[0])
    return result

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter
