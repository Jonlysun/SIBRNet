import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

def smooth_loss(depth,image):
    
    def gradient(pred):
        D_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        D_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        return D_dx, D_dy

    gradient_depth_x, gradient_depth_y = gradient(depth)

    gradient_img_x, gradient_img_y = gradient(image)

    exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x),1,True)) # (TODO)shape: bs,1,h,w
    exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y),1,True)) 

    smooth_x = gradient_depth_x*exp_gradient_img_x
    smooth_y = gradient_depth_y*exp_gradient_img_y

    return torch.mean(torch.abs(smooth_x))+torch.mean(torch.abs(smooth_y))

def DSSIM(x, y):
    # TODO: padding depend on the size of the input image sequences
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    # sigma_x = avepooling2d((x-mu_x)**2)
    # sigma_y = avepooling2d((y-mu_y)**2)
    # sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    sigma_x = avepooling2d(x**2)-mu_x**2
    sigma_y = avepooling2d(y**2)-mu_y**2
    sigma_xy = avepooling2d(x*y)-mu_x*mu_y
    k1_square = 0.01**2
    k2_square = 0.03**2
    # L_square = 255**2
    L_square = 1
    SSIM_n = (2*mu_x*mu_y+k1_square*L_square)*(2*sigma_xy+k2_square*L_square)
    SSIM_d = (mu_x**2+mu_y**2+k1_square*L_square) * \
        (sigma_x+sigma_y+k2_square*L_square)
    SSIM = SSIM_n/SSIM_d
    loss = torch.clamp((1-SSIM)/2, 0, 1)
    return loss

def image_similarity(alpha,x,y):
    return alpha*DSSIM(x,y)+(1-alpha)*torch.abs(x-y)
 

 
class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, es, ta, sr_result=None, vmm_map=None, x_feat_3_vmm=None):
        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        loss = []

        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1.
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return sum(loss)
