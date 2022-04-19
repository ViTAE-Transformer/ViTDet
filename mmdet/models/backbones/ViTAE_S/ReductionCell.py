import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer

class PRM(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12], in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.patch_shape = (self.img_size[0] // downsample_ratio, self.img_size[1] // downsample_ratio)

        self.convs = nn.ModuleList()
        for dilation in self.dilations:
            padding = math.ceil(((self.kernel_size-1)*dilation + 1 - self.stride) / 2)
            self.convs.append(nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                stride=self.stride, padding=padding, dilation=dilation),
                nn.GELU()]))

        self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, H, W = x.shape

        # to adapt to cases can be not divided
        padding = math.ceil(((self.kernel_size-1)*self.dilations[0] + 1 - self.stride) / 2)
        padding = [padding, padding]
        extra_padding = math.ceil(self.stride / 2) - 1
        wP = False
        hP = False
        if H % self.downsample_ratio != 0:
            hP = True
            padding[0] = padding[0] + extra_padding
        if W % self.downsample_ratio != 0:
            wP = True
            padding[1] = padding[1] + extra_padding

        y = F.gelu(F.conv2d(x, self.convs[0][0].weight, self.convs[0][0].bias, self.stride, tuple(padding), self.dilations[0])).unsqueeze(dim=-1)
        for i in range(1, len(self.dilations)):
            padding = math.ceil(((self.kernel_size-1)*self.dilations[i] + 1 - self.stride) / 2)
            padding = [padding, padding]
            if hP:
                padding[0] = padding[0] + extra_padding
            if wP:
                padding[1] = padding[1] + extra_padding
            _y = F.gelu(F.conv2d(x, self.convs[i][0].weight, self.convs[i][0].bias, self.stride, tuple(padding), self.dilations[i])).unsqueeze(dim=-1)
            y = torch.cat((y, _y), dim=-1)

        B, C, H, W, N = y.shape
        y = y.permute(0,4,1,2,3).flatten(3).reshape(B, N*C, W*H).permute(0,2,1).contiguous()

        return y, (H, W)

class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1, 2, 3, 4], tokens_type='performer', group=1,
                 attn_drop=0., drop_path=0., mlp_ratio=1.0, window_size=(14, 14)):
        super().__init__()

        self.img_size = img_size
        self.window_size = window_size
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        PCMStride = []
        residual = downsample_ratios // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None
        self.tokens_type = tokens_type
        if tokens_type == 'pooling':
            PCMStride = [1, 1, 1]
            self.pool = nn.MaxPool2d(downsample_ratios, stride=downsample_ratios, padding=0)
            tokens_type = 'transformer'
            downsample_ratios = 1

        self.PCM = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1), groups=group),  # the 1st convolution
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1), groups=group),  # the 1st convolution
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, token_dims, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1), groups=group),  # the 1st convolution
            nn.SiLU(inplace=True))


        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
            in_chans=in_chans, embed_dim=embed_dims)

        in_chans = self.PRM.out_chans

        self.patch_shape = self.PRM.patch_shape

        if tokens_type == 'performer':
            self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)
        elif tokens_type == 'performer_less':
            self.attn = None
            self.PCM = None
        elif tokens_type == 'transformer':
            self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, 
                                          attn_drop=attn_drop, drop_path=drop_path)

    def forward(self, x, size):
        H, W = size
        if len(x.shape) < 4:
            B, N, C  = x.shape
            x = x.reshape(B, H, W, C).contiguous()
            x = x.permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)

        shortcut = x
        PRM_x, _ = self.PRM(x)

        H, W = math.ceil(H / self.downsample_ratios), math.ceil(W / self.downsample_ratios)
        B, N, C = PRM_x.shape

        assert N == H * W, f"N is {N}, H is {H}, W is {W}, {shortcut.shape}"

        if self.attn is None:
            return PRM_x, (H, W)

        convX = self.PCM(shortcut)

        x = self.attn.attn(self.attn.norm1(PRM_x))
        convX = convX.permute(0, 2, 3, 1).reshape(*x.shape).contiguous()
        x = x + self.attn.drop_path(convX)
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))

        return x, (H, W)