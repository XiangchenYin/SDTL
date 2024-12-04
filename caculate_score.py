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
import cv2
import metric
from torchvision import transforms
import lpips

transform = transforms.Lambda(lambda t: (t * 2) - 1)

# test code:ssim, psnr, LPIPS



def test(generate_result_path, gt_path):

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
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    generate_result_path = 'results/LOLv1'
    gt_path = 'data/LOLv1/val/high'
    avg_psnr, avg_ssim = test(generate_result_path, gt_path)
