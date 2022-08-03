This is the official implementation of the paper [Learning Robust Image-Based Rendering on Sparse Scene Geometry via Depth Completion](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Learning_Robust_Image-Based_Rendering_on_Sparse_Scene_Geometry_via_Depth_CVPR_2022_paper.pdf), CVPR 2022.


# Setup
We use NVIDIA RTX 3090, cuda 11.1 and the follwing python packages:
- pytorch=1.7.0+cu110
- torchvision=0.8.2+cu110
- visdom=0.1.8.9
- tqdm

We borrows forward warping from [Forward warping](https://github.com/lizhihao6/Forward-Warp) and deformable convolution from [DCNv2](https://github.com/MatthewHowe/DCNv2/tree/master/DCN). Please follow their instruction to build this extension.


# Dataset
You can download our proposed dataset Surround from [here](https://drive.google.com/file/d/1h-8t4-iHLa3ujwUgcy7wmmgyXAeEORue/view?usp=sharing).

We also post-process two public dataset Tanks_and_Temples and Free View Synthesis, you can download them form [Tanks_and_Temples(comming soon)](), [Free_View_Synthesis(comming soon)]().

# Pre-trained Model'
You can download our pretrained dcnet and SIBRNet model from [dcnet](https://drive.google.com/file/d/1f22UsHubCkqdASt5G98V9-IlNBofeMXY/view?usp=sharing), [SIBRNet](https://drive.google.com/file/d/1xyjrVTN6mZ0RCax5fMD-_YheaXKQK8fw/view?usp=sharing). Pleace them in 'output'.

# Train

We use visdom to visualize the training process, please make sure it is connected. You can use 'python -m visdom.server --port xxxx' to build connection and open the visual interface in browser. Model will be saved in './outout'

## Depth Completion Net
Please train the depth completion net firstly.
```
python train_dc.py --prefix DCNet --visdom_port xxxx
```

## SIBRNet 
Train the full SIBRNet with:
```
python train_full.py --prefix SIBRNet --visdom_port xxxx
```

# Eval
```
python evaluation.py --prefix SIBRNet --batch_size 1 --visualize
```
You can use metrics_evaluate.py to calculate PSNR, SSIM, LPIPS with:
```
python metrcis_evaluate.py --dataset TanksSparse --sparse 4 --result_name 'Your result saved path'
```

# Citation
If you find this code useful, please cite our paper:
```
@inproceedings{sun2022learning,
  title={Learning Robust Image-Based Rendering on Sparse Scene Geometry via Depth Completion},
  author={Sun, Yuqi and Zhou, Shili and Cheng, Ri and Tan, Weimin and Yan, Bo and Fu, Lang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7813--7823},
  year={2022}
}
```

