import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.models import vgg16
import utils
import cv2, kornia
from models.unet import DiffusionUNet
from models.DiT import *
from models.SDTL import SDTL

from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM, Enhance
from accelerate import Accelerator
import time
from collections import OrderedDict
from IQA_pytorch import SSIM, MS_SSIM
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import pandas as pd

from caculate_score import test
logger = get_logger(__name__)

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

"""
用于生成扩散模型中的β值序列。在扩散模型中，β值代表了添加到数据的噪声水平，随时间步（num_diffusion_timesteps）逐渐变化。
函数根据输入参数 beta_schedule、beta_start、
beta_end 和 num_diffusion_timesteps 来计算出每个时间步对应的 β 值。
"""
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device
        # self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        # self.high_enhance1 = HFRM(in_channels=3, out_channels=64)

        self.enhance0 = Enhance(in_channels=3, out_channels=64)
        self.enhance1 = Enhance(in_channels=3, out_channels=64)

        print('model_type: ', self.config.model.model_type)

        if self.config.model.model_type == 'DDPM':
            self.Unet = DiffusionUNet(config)
        
        elif self.config.model.model_type == 'DiT':
            self.DiT = DiT(depth=4, hidden_size=384, patch_size=4, num_heads=12,
                  input_size=64, in_channels=6, learn_sigma=False)
    
        elif self.config.model.model_type == 'SDTL':
            # 模型总参数量: 24.84 M
            self.SDTL = SDTL(depth=6, hidden_size=384, patch_size=4, num_heads=6, mask_ratio=None, decode_layer=2,
              input_size=64, in_channels=6, learn_sigma=False)
        
        for name, param in self.named_parameters():
            param.requires_grad = True            
        
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_mask(self, dark):

        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        return torch.cat([mask, mask, mask], dim=1)

    def compute_local_entropy(self, x, kernel_size=3, num_bins=256):
        assert x.dim() == 4, 'Input must be a 4D tensor'

        b, c, h, w = x.size()

        # Create bin edges
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=x.device)

        # Create a sliding window view of the input tensor
        x_unfolded = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
        x_unfolded = x_unfolded.view(b, c, -1, h, w)

        # Initialize local histogram tensor
        local_histogram = torch.zeros(b, c, num_bins, h, w, device=x.device)

        # Compute the local histogram using histc
        for i in range(num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            mask = (x_unfolded >= lower) & (x_unfolded < upper)
            local_histogram[:, :, i] = torch.sum(mask, dim=2)

        # Normalize the local histogram
        local_histogram = local_histogram / (kernel_size * kernel_size)

        # Compute the local entropy
        local_entropy = -torch.sum(local_histogram * torch.log2(local_histogram + 1e-9), dim=2)

        return local_entropy

    def get_condition(self, x):
        imgs = x

        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        imgs = imgs * 255
        imgs = imgs.astype(np.uint8)

        canny_edges = []
        for img in imgs:
            # 对每个图片进行Canny边缘检测
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            edges = cv2.Canny(gray_img, threshold1=10, threshold2=40)  # Canny边缘检测，阈值需根据实际情况调整
            edges = HWC3(edges)

            canny_edges.append(edges)
        
        condition = torch.from_numpy(np.stack(canny_edges)).permute(0, 3, 1, 2).float() 

        condition = condition.to(self.device)
        return condition

    def sample_training(self, x_cond, b, x, input_img=None, eta=0., enable_mask=False): 
        # x_cond:小波变换后的img_norm, b:噪声水平, eta:噪声比例
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1]) 
        x = torch.randn(n, c, h, w, device=self.device) # 与x_cond相同大小的高斯噪声x
        xs = [x] 
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            if self.config.model.model_type == 'DDPM':
                et = self.Unet(torch.cat([x_cond, xt], dim=1), t) # x_cond: torch.Size([8, 3, 64, 64]); xt: torch.Size([8, 3, 64, 64])
            elif self.config.model.model_type == 'DiT':
                et = self.DiT(torch.cat([x_cond, xt], dim=1), t)
            elif self.config.model.model_type == 'SDTL':
                condition = self.get_condition(input_img)
                et = self.SDTL(torch.cat([x_cond, xt], dim=1), t, condition=condition, enable_mask=enable_mask)
            elif self.config.model.model_type == 'SDiT':
                et = self.SDiT(torch.cat([x_cond, xt], dim=1), t)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape # 8 3 256 256
        input_img_norm = data_transform(input_img) # 8 3 256 256
        # 通过DWT（离散小波变换）和IWT（逆离散小波变换）对输入图像进行处理
        input_dwt = dwt(input_img_norm)  # 32, 3, 128, 128

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...] # 8, 3, 128, 128 | 24, 3, 128, 128
        condition = self.get_condition(input_img)

        input_high0 = self.enhance0(input_high0, condition) # torch.Size([24, 3, 128, 128])

        input_LL_dwt = dwt(input_LL) 
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...] # 8, 3, 64, 64 | 24, 3, 64, 64
        input_high1 = self.enhance1(input_high1, condition) # 24, 3, 64, 64

        b = self.betas.to(input_img.device)
        # 变量b代表了一组预先定义好的beta值（β），
        # 通常在去噪过程中用于控制噪声的添加或去除程度。

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        # 根据给定的随机时间步长生成一组系数，
        # 这些系数将在去噪或生成过程中控制每个步骤的处理程度。

        e = torch.randn_like(input_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            # 使用Unet模型（self.Unet）结合加噪后的低频部分x以及原始输入的低频子带input_LL_LL，
            # 根据给定的时间步长t.float()产生噪声输出noise_output。
            if self.config.model.model_type == 'DDPM':
                noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())
            elif self.config.model.model_type == 'DiT':
                noise_output = self.DiT(torch.cat([input_LL_LL, x], dim=1), t.float())
            elif self.config.model.model_type == 'SDTL':
                condition = self.get_condition(input_img)
                noise_output = self.SDTL(torch.cat([input_LL_LL, x], dim=1), t.float(), condition = condition, enable_mask=True)
            
            denoise_LL_LL = self.sample_training(input_LL_LL, b, x, input_img, enable_mask=True)
            # 调用sample_training方法基于输入低频子带input_LL_LL和beta值b来获取去噪后的低频子带denoise_LL_LL。

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b, x, input_img, enable_mask=False)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config, DATASET):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.model = Net(args, config)
        self.model.to(self.device)
        
        if args.accelerator_train:
            kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(kwargs_handlers=[kwargs], 
                split_batches=True, step_scheduler_with_optimizer=False)
            
            self.device = self.accelerator.device
            self.model.to(self.accelerator.device)

        if os.path.isfile(self.args.resume) and args.accelerator_train:
            print(f"--------Resumed from checkpoint: {args.resume}")
            self.start_epoch, self.step = 0, 0
            self.load_ddm_ckpt(self.args.resume, ema=False)

        else:
            self.start_epoch, self.step = 0, 0

        self.train_loader, self.val_loader = DATASET.get_loaders()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())

        if args.accelerator_train:
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()
        self.MSSIM_loss = MS_SSIM()

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        print(checkpoint['config'])
        state_dict = OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            if 'module.' in key:
                # 去掉'module.'前缀
                new_key = key.replace('module.', '')
            else:
                new_key = key
            state_dict[new_key] = value

        self.model.load_state_dict(state_dict, strict=True)

        # Trainable copy weights load
        if self.config.model.model_type == 'Control' and self.config.model.control_stage == 2:
            selected_layers_dict = {k.replace('controldm.diffusion_model.', ''): v for k, v in state_dict.items() if 'controldm.diffusion_model' in k}  
            self.model.controldm.control_model.load_state_dict(selected_layers_dict, strict=False)

        # state_dict = OrderedDict()
        # for key, value in checkpoint['ema_helper'].items():
        #     if 'module.' in key:
        #         # 去掉'module.'前缀
        #         new_key = key.replace('module.', '')
        #     else:
        #         new_key = key
        #     state_dict[new_key] = value

        # self.ema_helper.load_state_dict(state_dict)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        # if ema:
        #     self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} epoch {} step {}".format(load_path, self.start_epoch, self.step))

    def train(self):
        cudnn.benchmark = True
        # if os.path.isfile(self.args.resume):
        #     self.load_ddm_ckpt(self.args.resume)
        best_psnr = 0
        best_ssim = 0
        epochs = []
        psnr = []
        ssim = []

        for epoch in range(self.start_epoch, self.config.training.n_epochs):

            if self.args.accelerator_train:
                self.accelerator.print('--------Start epoch: ', epoch)
            else:
                print('--------Start epoch: ', epoch)

            iters = len(self.train_loader)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(self.train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1
                if not self.args.accelerator_train:
                    x= x.to(self.device)

                output = self.model(x)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)


                loss = noise_loss + photo_loss + frequency_loss

                if self.step % 10 == 0:

                    if self.args.accelerator_train:
                        self.accelerator.print("epoch:{}, step:{}, Curr_epoch_datatime:{}, lr:{:.6f}, loss:{:.4f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                            "frequency_loss:{:.4f}".format(epoch, self.step,
                                                           data_time,
                                                           self.scheduler.get_last_lr()[0],
                                                           loss.item(),
                                                           noise_loss.item(), photo_loss.item(),
                                                           frequency_loss.item()  ))
                    else:
                        print("epoch:{}, step:{}, Curr_epoch_datatime:{}, lr:{:.6f}, loss:{:.4f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                            "frequency_loss:{:.4f}".format(epoch, self.step,
                                                           data_time,
                                                           self.scheduler.get_last_lr()[0],
                                                           loss.item(),
                                                           noise_loss.item(), photo_loss.item(),
                                                           frequency_loss.item()  ))

                self.optimizer.zero_grad()

                if self.args.accelerator_train:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                self.optimizer.step()
                
                self.ema_helper.update(self.model)
                data_start = time.time()

            if epoch % self.config.training.save_epoch == 0 and epoch != 0:
                self.model.eval()
                self.accelerator.print(f"Save weights to {os.path.join(self.args.image_folder, f'model_latest_epoch_{epoch}')}")
                if self.args.accelerator_train:
                    self.accelerator.wait_for_everyone()  # 可以阻塞所有先到达的进程，直到所有其他进程都达到了相同的点
                    if self.accelerator.is_main_process:
                        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                       'state_dict': self.model.state_dict(),
                                                       'optimizer': self.optimizer.state_dict(),
                                                       'scheduler': self.scheduler.state_dict(),
                                                       'ema_helper': self.ema_helper.state_dict(),
                                                       'params': self.args,
                                                       'config': self.config},
                                                      filename=os.path.join(self.args.image_folder,
                                                                            'model_latest'))

                        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                       'state_dict': self.model.state_dict(),
                                                       'optimizer': self.optimizer.state_dict(),
                                                       'scheduler': self.scheduler.state_dict(),
                                                       'ema_helper': self.ema_helper.state_dict(),
                                                       'params': self.args,
                                                       'config': self.config},
                                                      filename=os.path.join(self.args.image_folder,
                                                                            f'model_epoch_{epoch}'))
                else:
                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.args.image_folder,
                                                                        'model_latest'))

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.args.image_folder,
                                                                        f'model_epoch_{epoch}'))

            if epoch % self.config.training.vis_sample_epoch == 0:

                self.model.eval()
                if self.args.accelerator_train:
                    self.accelerator.wait_for_everyone()  # 可以阻塞所有先到达的进程，直到所有其他进程都达到了相同的点
                    if self.accelerator.is_main_process:
                        avg_psnr, avg_ssim = self.sample_validation_patches(self.val_loader, epoch)

                        epochs.append(epoch)
                        psnr.append(avg_psnr)
                        ssim.append(avg_ssim)
                        data = pd.DataFrame({'Epoch': epochs, 'PSNR': psnr, 'SSIM': ssim})
                        data.to_csv(os.path.join(self.args.image_folder, 'psnr_ssim.csv'), index=False)

                        plot_psnr_ssim_trends(epochs, psnr, ssim, os.path.join(self.args.image_folder, 'psnr_ssim.png'))

                        if best_psnr < avg_psnr:
                            best_psnr = avg_psnr    
                            best_ssim = avg_ssim
                            
                            utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.args.image_folder,
                                                                        f'model_best_epoch'))
                        print(f'The best psnr is {best_psnr}, The best ssim is {best_ssim}')

                else:
                    avg_psnr, avg_ssim = self.sample_validation_patches(self.val_loader, epoch)
                    if best_psnr < avg_psnr:
                            best_psnr = avg_psnr
                            utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.args.image_folder,
                                                                        f'model_best_epoch'))

            # 这条线是每个epoch结束                             
            self.scheduler.step()
        # if self.args.accelerator_train:
        #     self.accelerator.end_training()

    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_LL))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)
        # ssim_loss = self.MSSIM_loss(pred_x, gt_img)
        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, epoch):
        image_folder = os.path.join(self.args.image_folder, 'image_log')
        self.model.eval()

        ssim = SSIM()
        psnr = PSNR()

        ssim_list = []
        psnr_list = []

        with torch.no_grad():
            print(f"Processing a single batch of validation images at epoch: {epoch}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                utils.logging.save_image(pred_x, os.path.join(image_folder, 'epoch_'+str(epoch), f"{y[0]}.png"))
            # 求psnr和ssim等指标，与测试的函数一致；
            gt_path = os.path.join(self.config.data.data_dir, self.config.data.gt_dir)
            generate_result_path = os.path.join(image_folder, 'epoch_'+str(epoch))
            avg_psnr, avg_ssim = test(generate_result_path, gt_path)
        return avg_psnr, avg_ssim

EPS = 1e-3
PI = 22.0 / 7.0
# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

import matplotlib.pyplot as plt
import pandas as pd

def plot_psnr_ssim_trends(epochs, psnr_values, ssim_values, save_fig_path):
    

    # 最近200个epoch的数据子集
    recent_epochs = epochs[-10:] if len(epochs) >= 10 else epochs
    recent_psnr = psnr_values[-10:] if len(psnr_values) >= 10 else psnr_values
    recent_ssim = ssim_values[-10:] if len(ssim_values) >= 10 else ssim_values


    # 计算最近10次的PSNR和SSIM最大值
    max_recent_psnr = max(recent_psnr)
    max_all_psnr = max(psnr_values)
    max_recent_ssim = max(recent_ssim)
    max_all_ssim = max(ssim_values)


    # 创建一个 figure 对象，并设置子图布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 绘制最近 200 次的 PSNR 和 SSIM
    axs[0, 0].plot(recent_epochs, recent_psnr, color='orange', linestyle='--')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('PSNR')
    axs[0, 0].set_title('Recent 200 Epoch PSNR')

    axs[0, 0].axhline(y=max_recent_psnr, color='blue', linestyle='-')
    axs[0, 0].annotate(f'Max: {max_recent_psnr:.2f}', 
                    xy=(recent_epochs[-1], max_recent_psnr), 
                    xytext=(recent_epochs[-1]+1, max_recent_psnr+0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05))


    axs[0, 1].plot(recent_epochs, recent_ssim, color='orange', linestyle='--')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].set_title('Recent 200 Epoch SSIM')


    axs[0, 1].plot(recent_epochs, recent_ssim, color='orange', linestyle='--')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].axhline(y=max_recent_ssim, color='blue', linestyle='-')
    axs[0, 1].annotate(f'Max: {max_recent_ssim:.4f}', 
                        xy=(recent_epochs[-1], max_recent_ssim), 
                        xytext=(recent_epochs[-1]+1, max_recent_ssim+0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05))

    # 绘制所有的 PSNR 和 SSIM
    axs[1, 0].plot(epochs, psnr_values, color='green', linestyle='--')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('PSNR')
    axs[1, 0].set_title('All Epoch PSNR')

    axs[1, 0].axhline(y=max_all_psnr, color='blue', linestyle='-')
    axs[1, 0].annotate(f'Max: {max_all_psnr:.2f}', 
                        xy=(epochs[-1], max_all_psnr), 
                        xytext=(epochs[-1]+1, max_all_psnr+0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05))

    axs[1, 1].plot(epochs, ssim_values, color='green', linestyle='--')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('SSIM')
    axs[1, 1].set_title('All Epoch SSIM')

    axs[1, 1].axhline(y=max_all_ssim, color='blue', linestyle='-')
    axs[1, 1].annotate(f'Max: {max_all_ssim:.4f}', 
                        xy=(epochs[-1], max_all_ssim), 
                        xytext=(epochs[-1]+1, max_all_ssim+0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # 保存图片
    plt.savefig(save_fig_path, dpi=300)

    # 可选：显示图像（如果在脚本执行环境中需要查看）
    # plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    @staticmethod
    def rgb_to_lab(img):
        assert img.dim() == 4, "Input image should be a 4D tensor (B, C, H, W)"
        assert img.size(1) == 3, "Input image should have 3 channels (RGB)"

        img = img.permute(0, 2, 3, 1)  # Change to (B, H, W, C) for processing

        # Convert to Lab color space
        img_lab = torch.zeros_like(img)
        img_lab[..., 0] = 0.412453 * img[..., 0] + 0.357580 * img[..., 1] + 0.180423 * img[..., 2]
        img_lab[..., 1] = 0.212671 * img[..., 0] + 0.715160 * img[..., 1] + 0.072169 * img[..., 2]
        img_lab[..., 2] = 0.019334 * img[..., 0] + 0.119193 * img[..., 1] + 0.950227 * img[..., 2]

        img_lab[..., 0] = (img_lab[..., 0] / 0.950456) ** (1 / 3) - 16 / 116
        img_lab[..., 1] = (img_lab[..., 1]) ** (1 / 3) - 16 / 116
        img_lab[..., 2] = (img_lab[..., 2] / 1.088754) ** (1 / 3) - 16 / 116

        img_lab[..., 1] = img_lab[..., 1] * 500
        img_lab[..., 2] = img_lab[..., 2] * 200

        img_lab = img_lab.permute(0, 3, 1, 2)  # Change back to (B, C, H, W)
        return img_lab

    def forward(self, img1, img2):
        assert img1.size() == img2.size(), "Input images should have the same shape"
        assert img1.size(1) == 3, "Input images should have 3 channels (RGB)"

        img1_lab = self.rgb_to_lab(img1)
        img2_lab = self.rgb_to_lab(img2)

        color_loss = F.l1_loss(img1_lab, img2_lab)
        return color_loss


# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)



