import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
import cv2
import metric
from torchvision import transforms
import lpips

transform = transforms.Lambda(lambda t: (t * 2) - 1)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='./LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='experiments/MDT_nomasked_patch4_img256_8V100_StepLR_SNRTransformerBlock_EntropyEnhance/model_epoch_100.pth.tar', 
                        type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results', type=str,
                        help="Location to save restored images")
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.type))
    DATASET = datasets.__dict__[config.data.type](config)
    # _, val_loader = DATASET.get_loaders(parse_patches=False)
    _, val_loader = DATASET.get_loaders()


    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config, DATASET)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader)

    # test code:ssim, psnr, LPIPS
    gt_path = os.path.join(config.data.data_dir, config.data.gt_dir)
    generate_result_path = os.path.join(args.image_folder, config.data.type)

    loss_fn_vgg = lpips.LPIPS(net='alex')
    lpipss = [] 
    avg_ssim = 0 
    avg_psnr = 0 

    for result in os.listdir(generate_result_path):
        gt_img = cv2.imread(os.path.join(gt_path, result))
        normal_img = cv2.imread(os.path.join(generate_result_path, result))

        gt_img = gt_img / 255.
        normal_img = normal_img / 255.
        mean_gray_out = cv2.cvtColor(normal_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_gray_gt = cv2.cvtColor(gt_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        normal_img_adjust = np.clip(normal_img * (mean_gray_gt / mean_gray_out), 0, 1)

        normal_img = (normal_img_adjust * 255).astype(np.uint8)
        gt_img = (gt_img * 255).astype(np.uint8)
        # print(normal_img.shape, gt_img.shape)
        psnr = metric.calculate_psnr(normal_img, gt_img)
        ssim = metric.calculate_ssim(normal_img, gt_img)

        # lpips
        img_hq = np.transpose(normal_img / 255, (2, 0, 1))
        img_hq = transform(torch.from_numpy(img_hq).unsqueeze(0))
        img_gt = np.transpose(gt_img / 255, (2, 0, 1))
        img_gt = transform(torch.from_numpy(img_gt).unsqueeze(0))
        lpips_ = loss_fn_vgg(img_hq.to(torch.float32), img_gt.to(torch.float32))

        # lpips_ = loss_fn_vgg(visuals['HQ'], visuals['GT'])
        lpipss.append(lpips_.detach().numpy())

        print(f'Single image {result} SSIM is {ssim}, PSNR is {psnr}, LPIPS is {lpips_.detach().numpy()[0][0][0][0]}')
        avg_ssim += ssim
        avg_psnr += psnr

    avg_psnr = avg_psnr / len(os.listdir(generate_result_path))
    avg_ssim = avg_ssim / len(os.listdir(generate_result_path))
    print('# Validation # avgPSNR: {} avgSSIM: {} avgLPIPS: {}'.format(avg_psnr, avg_ssim, np.mean(lpipss)))

if __name__ == '__main__':
    main()
