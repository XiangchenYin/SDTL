import math
import torch
import torch.nn as nn
import torch.nn.functional
from abc import abstractmethod
from models.unet import *
import kornia
import numpy as np
import cv2

class ProcessBlock(nn.Module):
    def __init__(self, in_ch):
        super(ProcessBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x):
        return x+self.block(x)


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # self.down_conv = nn.Sequential(
        #     conv_nd(2, self.ch, self.ch, 3, padding=1, stride=2),
        #     nn.SiLU(),
        # )

        self.down_conv = nn.Sequential(
            conv_nd(2, self.ch, self.ch, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, self.ch, self.ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, self.ch, self.ch, 3, padding=1, stride=2),
        )

        self.conn_convs = nn.ModuleList([self.make_conv(self.ch)])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
    
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            
            block.append(conv_nd(2, block_in, block_out, 1, padding=0))

            block_in = block_out

            # block.append(conv_nd(2, block_in, block_out, 1, padding=0))
            block.append(ProcessBlock(block_in))
            # self.conn_convs.append(self.make_conv(block_out))
            self.conn_convs.append(self.make_conv(block_out))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = conv_nd(2, block_in, block_in, 1, padding=0)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = conv_nd(2, block_in, block_in, 1, padding=0)

        self.middle_block_out = self.make_conv(block_in)

    def make_conv(self, channels):
        return conv_nd(2, channels, channels, 1, padding=0)
    
    def forward(self, x):

        outs = []
        # downsampling
        hs = [self.down_conv(self.conv_in(x))]

        i = 0
        outs.append(self.conn_convs[i](hs[0]))
        i += 1

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):              
                h = self.down[i_level].block[i_block](hs[-1])

                hs.append(h)
            outs.append(self.conn_convs[i](h))
            i+=1 

            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        outs.append(self.middle_block_out(h))
        return outs

class AdapterUnet(DiffusionUNet):

    def __init__(self, config):
        super().__init__(config)
    
    # 本文UNet的forward
    def forward(self, x, t, control=None):
        # assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        if control is not None:
            h += control.pop()
        
        # 需要明确zero-conv的数量，每个res block or 每个 block???
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):

                if i_block==0 and control is not None:
                    h = torch.cat([h, hs.pop() + control.pop()], dim=1)
                else:
                    h = torch.cat([h, hs.pop()], dim=1)

                h = self.up[i_level].block[i_block](
                    h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class AdapterDM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.adapter = Adapter(config)
        self.diffusion_model = AdapterUnet(config)

    def forward(self, x, condition, t):

        control = self.adapter(condition) # Control是用xt还是x_cond和xt的拼接？？？
        et = self.diffusion_model(x, t, control=control)
        return et


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # elif isinstance(layer, SpatialTransformer):
            #     x = layer(x, context)
            else:
                x = layer(x)
        return x


