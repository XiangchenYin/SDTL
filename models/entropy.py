import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2



class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class Entropy(nn.Module):
    def __init__(self, dim):
        super(Entropy, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.local_entropy = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            )
        
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        
        self.condtion_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            )

        # self.global_entropy = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, condition=None):

        b, c, h, w = x.shape
        # import time; start = time.time()
        # entropy = self.compute_local_entropy( torch.sum(x, dim=1).unsqueeze(1), 
        #                    kernel_size=3) 
        
        # # print(time.time()-start, entropy.shape)
        # entropy = self.local_entropy(entropy) 

        if condition is not None:
            condition = F.interpolate(condition, size=(h,w), mode='bilinear', align_corners=True)
        condition = self.condtion_conv(condition)
        # print(condition.shape, x.shape)
        # Reshape local entropy from (b, c, h, w) to (b, c, h*w)
        # entropy = entropy.view(b, 1, -1)

        # # Apply softmax to the local entropy
        # entropy_weights = self.softmax(entropy)

        # # Reshape attention weights from (b, c, h*w) to (b, c, h, w)
        # entropy_weights = entropy_weights.view(b, 1, h, w)

        enhance_x = condition + x
        
        atten = self.spatial(enhance_x)
        atten = self.sigmoid(atten)

        output = x * atten + x
        return output        


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