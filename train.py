import argparse
import os
import random
import shutil
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='./configs/LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='', type=str,
                        help="Location to save restored validation image and model during training")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    parser.add_argument(
        "--accelerator_train",
        action="store_true",
        help="Whether the multi-GPU train in accelerator."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    # setup device to run
    if not args.accelerator_train:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device: {}".format(device))
        config.device = device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config.device = device

        print('Use the accelerator training, Device:')
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU #{i}: {device_name}")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.type))
    DATASET = datasets.__dict__[config.data.type](config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config, DATASET)
    diffusion.train()


if __name__ == "__main__":
    main()
