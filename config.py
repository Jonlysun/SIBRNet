import os
import numpy as np
import random
import yaml
import argparse

HOME_PATH = '/home/user2/SIBRNet'

Tanks_and_Temples_root = '/home/user2/Datasets/Tanks_and_Temples_CVPR'
Tank_scenes = [
    'Train', 
    'M60', 
    'Playground', 
    'Truck'
]

Surround_root = '/home/user2/Datasets/Surround_CVPR'
Surround_scenes = [
    'basketball',
    'meetingroom',
    'park',
    'philosopher',
    'soccer',
    'statue'
]

FVS_root = '/home/user2/Datasets/FreeViewSynthesis_CVPR'
FVS_own_scenes = [
    "pirate",
    "playground",
    "bike",
    "flowers",
    "sandbox",
    "soccertable"
]

def argparser(is_train=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--dataset', type=str,
                        default='TanksSparse', choices=['TanksSparse', 'FVS_own_Sparse', 'Surround'])
    parser.add_argument('--num_input', type=int, default=5)
    parser.add_argument('--scale', type=float,
                        default=0.25, choices=[0.5, 0.25])
    parser.add_argument('--sparse', type=str,
                        default='4', choices=['4', '8', '16', 'all'])
    parser.add_argument('--step', type=int,
                        default=1, choices=[1, 2, 3])
    parser.add_argument('--tanks_train_nbs_mode', type=str,
                        default='near', choices=['near', 'argmax', 'near1'])                        
    parser.add_argument('--max_depth', type=float, default=255.0,
                        help='maximum depth')
    parser.add_argument('--prefix', type=str, default='',
                        help='a nickname for the training')
    parser.add_argument('--save_dir', type=str,
                        default='output', help='directory to save the model')
    parser.add_argument('--model_checkpoint', type=str,
                        default='model', help='model')
    parser.add_argument('--optimizer_checkpoint', type=str,
                        default='training', help='optimizer')

    # training
    parser.add_argument('--patch_height', type=int,
                        default=256, help='patch for training')
    parser.add_argument('--patch_width', type=int,
                    default=256, help='patch for training')
    parser.add_argument('--max_epoch', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_steps', type=list, default=[30, 35])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--train_device', type=str, default='cuda:0')
    parser.add_argument('--resume_epoch', type=int)
    parser.add_argument('--mix', type=str, default='nomix', choices=['mix', 'nomix'])

    # Architecture
    parser.add_argument('--DCnet', type=str, default='dcnet')
    parser.add_argument('--DCnet_restore_epoch', type=int, default=59)
    parser.add_argument('--DCnet_ft', action='store_true', default=False)
    parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--resnet_pretrained', action='store_true', default=True)

    # testing
    parser.add_argument('--restore_epoch', type=int, default=39)
    parser.add_argument('--eval_device', type=str, default='cuda:0')
    parser.add_argument('--metric', type=str, default='Multi', choices=['LPIPS', 'Multi'])
    parser.add_argument('--eval_seq', type=str, default='all', choices=['all', 'sub'])
    parser.add_argument('--visualize', action="store_true", default=False)

    config = parser.parse_args()

    # save config
    save_dict = config.__dict__
    save_path = os.path.join(config.save_dir, config.prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, 'config.yaml')
    with open(save_file, 'w', encoding='utf-8') as f:
        yaml.dump(save_dict, f)

    return config

if __name__ == '__main__':
    config = argparser(is_train=False)
    print(config.restore_epoch)
