import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import trunc_normal_
import math
import random
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.rel_pos_bias = RelativePositionBias(
        #     window_size=[int(num_patches**0.5), int(num_patches**0.5)], num_heads=num_heads)

    def get_masked_rel_bias(self, B, ids_keep):
        # get masked rel_pos_bias
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias.unsqueeze(dim=0).repeat(B, 1, 1, 1)

        rel_pos_bias_masked = torch.gather(
            rel_pos_bias, dim=2, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, rel_pos_bias.shape[1], 1, rel_pos_bias.shape[-1]))
        rel_pos_bias_masked = torch.gather(
            rel_pos_bias_masked, dim=3, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, rel_pos_bias.shape[1], ids_keep.shape[1], 1))
        return rel_pos_bias_masked

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if ids_keep is not None:
        #     rp_bias = self.get_masked_rel_bias(B, ids_keep)
        # else:
        #     rp_bias = self.rel_pos_bias()
        
        # attn += rp_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(
                size=(window_size[0] * window_size[1],) * 2, dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index",
                             relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        labels = torch.where(drop_ids.to(labels.device),
                             self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings




#################################################################################
#                                 Core MDT Model                                #
#################################################################################

class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, c, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), ids_keep=ids_keep)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of MDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# class AdaptiveFusion(nn.Module):
#     def __init__(self, embed_dim):
#         super(AdaptiveFusion, self).__init__()
#         self.fc_x = nn.Linear(embed_dim, 1)
#         self.fc_cond = nn.Linear(embed_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(embed_dim*2, embed_dim)

#     def forward(self, x_embed, cond_embed):
#         # 计算加权系数
#         alpha_x = self.sigmoid(self.fc_x(x_embed))
#         alpha_cond = self.sigmoid(self.fc_cond(cond_embed))

#         # 对两个嵌入信息进行融合
#         fused_embed = torch.cat([alpha_x * x_embed, alpha_cond * cond_embed], dim = 2)
#         out = self.proj(fused_embed)
#         return out

class AdaptiveFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AdaptiveFusion, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att = nn.Linear(embed_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 计算加权系数
        x_embed = self.att(x) * x  + x
        global_x_embed = self.sigmoid(self.conv1(self.avg_pool(x))) * x_embed

        out = self.proj(global_x_embed)
        return out



class AttenScore(nn.Module):
    def __init__(self, dim):
        super().__init__()
    
        self.mlp = Mlp(
        in_features=dim,
        hidden_features=dim*2,
        out_features=1
        )

    def forward(self, tokens):
        scores = self.mlp(tokens).squeeze(-1)
        attention_weights = F.sigmoid(scores)  # 对每个序列应用softmax
        return attention_weights




class GuideAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads):
        super(GuideAttentionModule, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
    def forward(self, x, tokens):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        tokens_Q = self.query(tokens)
        tokens_K = self.key(tokens)

        Q = Q.view(Q.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        tokens_Q = tokens_Q.view(tokens_Q.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        tokens_K = tokens_K.view(tokens_K.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine SNR query and key with the original query and key
        combined_Q = Q + tokens_Q
        combined_K = K + tokens_K

        attn_scores = torch.matmul(combined_Q, combined_K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.d_model)

        return attn_output


class GuideTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_hidden_dim):
        super(GuideTransformerBlock, self).__init__()

        self.attention = GuideAttentionModule(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, tokens):
        # Attention layer
        attn_output = self.attention(x, tokens)
        x = x + attn_output
        x = self.layer_norm1(x)

        # MLP layer
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.layer_norm2(x)

        return x


class CAFusion(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CAFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

    def forward(self, tokens1, tokens2):
        # tokens1的尺寸为(batch_size, seq_len1, feature_dim)
        # tokens2的尺寸为(batch_size, seq_len2, feature_dim)

        batch_size = tokens1.size(0)

        # 计算query, key, value
        query = self.query(tokens1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(tokens2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(tokens2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # 计算加权和
        weighted_values = torch.matmul(attention_weights, value)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # 输出线性层
        fused_tokens = self.out(weighted_values)

        return fused_tokens

class SDTL(nn.Module):
    """
    Struture-Guide Diffusion Transformer for Low-Light Enhancement
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        mask_ratio=None,
        decode_layer=2,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False)
        
        # self.x_embedder2 = PatchEmbed(
        #     input_size, patch_size*2, in_channels, hidden_size, bias=True, strict_img_size=False)
        self.cond_embedder = PatchEmbed(
            input_size, patch_size*4, in_channels//2, hidden_size, bias=True, strict_img_size=False)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        # Will use learnbale sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(
        #     1, num_patches, hidden_size), requires_grad=True)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) for _ in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(1)
        ])
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels)

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(
        #     1, num_patches, hidden_size), requires_grad=True)
        
        # self.align = nn.ConvTranspose1d(in_channels=hidden_size, out_channels=hidden_size, 
        #                         kernel_size=4, stride=4)
        
        # self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, 
        #                         kernel_size=4, stride=4)
        
        # self.fuse = AdaptiveFusion(embed_dim=hidden_size)

        self.guide_blocks = nn.ModuleList([
            GuideTransformerBlock(d_model=hidden_size,
         num_heads=num_heads, mlp_hidden_dim=hidden_size*2)
            for _ in range(half_depth*2)
        ])

        # self.cond_fuse = CAFusion(feature_dim=hidden_size,num_heads=8)
        # self.atten_score = AttenScore(dim=hidden_size)
        # self.low_block = MDTBlock(hidden_size, num_heads, 
        #             mlp_ratio=mlp_ratio, num_patches=num_patches)

        # self.proj1 = nn.Linear(hidden_size*2, hidden_size)
        # self.proj2 = nn.Linear(hidden_size*2, hidden_size)

        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(torch.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.initialize_weights()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(
        #     torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.decoder_pos_embed.data.copy_(
        #     torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.en_inblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.en_outblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.de_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # if self.mask_ratio is not None:
        #     torch.nn.init.normal_(self.mask_token, std=.02)

    def unpatchify(self, x, x_shape):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """ 
        # 28800
        # 
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x_shape[2]//p
        w = x_shape[3]//p
        # h = w = int(x.shape[1] ** 0.5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        # x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def split_tokens_by_scores(self, tokens, scores, split_ratio=0.8):
        # 对分数进行排序，并获取排序后的索引
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)

        # 计算高分和低分token的数量
        num_tokens = tokens.shape[1]
        num_high_score_tokens = int(num_tokens * split_ratio)
        num_low_score_tokens = num_tokens - num_high_score_tokens

        # 使用排序后的索引将原始token序列分割成高分和低分序列
        high_score_tokens = torch.gather(tokens, 1, sorted_indices[:, :num_high_score_tokens].unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        low_score_tokens = torch.gather(tokens, 1, sorted_indices[:, num_high_score_tokens:].unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        return high_score_tokens, low_score_tokens
    def forward(self, x, t, condition = None, enable_mask=True):
        """
        Forward pass of MDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        enable_mask: Use mask latent modeling
        """
        x_shape = x.shape 
        # x2 = self.x_embedder2(x)
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        c = t                             # (N, D)

        if condition is not None:
            cond = self.cond_embedder(condition)
        
        # print(self.mask_ratio, enable_mask, end='')
        input_skip = x

        masked_stage = False
        skips = []
        skips2 = []
        # masking op for training
        if self.mask_ratio is not None and enable_mask:
            # print('masking: length -> length * mask_ratio')
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True
        
        i = 0
        for block in self.en_inblocks:
            if masked_stage:
                x = block(x, c, ids_keep=ids_keep)
            else:
                x = block(x, c, ids_keep=None)
                # x2 = block(x2, c, ids_keep=None)
                x = self.guide_blocks[i](x, cond)
                i = i + 1


            skips.append(x)
            # skips2.append(x2)

        # print('skip', [i.shape for i in skips])
        # print('input_skip', input_skip.shape)
        for block in self.en_outblocks:
            if masked_stage:
                x = block(x, c, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = block(x, c, skip=skips.pop(), ids_keep=None)
                # x2 = block(x2, c, skip=skips2.pop(), ids_keep=None)

                x = self.guide_blocks[i](x, cond)
                i = i + 1

        if self.mask_ratio is not None and enable_mask:
            x = self.forward_side_interpolater(x, c, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            # x = x + self.decoder_pos_embed
            x = x
        
        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip
            x = block(x, c, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, c) 
        x = self.unpatchify(x, x_shape)      # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0):
        """
        Forward pass of MDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if cfg_scale is not None:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            scale_step = (
                1-torch.cos(((1-t/diffusion_steps)**scale_pow)*math.pi))*1/2 # power-cos scaling 
            real_cfg_scale = (cfg_scale-1)*scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)

            half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            model_out = self.forward(x, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   MDTv2 Configs                               #
#################################################################################

def MDTv2_XL_2(**kwargs):
    return MDTv2(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MDTv2_L_2(**kwargs):
    return MDTv2(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MDTv2_B_2(**kwargs):
    return MDTv2(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MDTv2_S_2(**kwargs):
    return MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

if __name__ == '__main__':
    
    mdt = MDTv2(depth=6, hidden_size=384, patch_size=4, num_heads=12, mask_ratio=None, decode_layer=2,
              input_size=64, in_channels=6, learn_sigma=False)

    x = torch.ones(4, 6, 80, 64)
    t = torch.ones(4, )
    print('mdt', mdt(x,t, enable_mask=False).shape)
