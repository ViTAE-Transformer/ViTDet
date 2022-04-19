import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
import torch.nn.functional as F
from mmdet.models.backbones.utils.WindowAttention import WindowAttention, calc_rel_pos_spatial

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.window_size = window_size
        q_size = window_size[0]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NormalCell(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=64, tokens_type='transformer', 
                img_size=224, window=False, window_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_size = img_size
        self.window_size = window_size

        self.tokens_type = tokens_type

        assert tokens_type == 'transformer'

        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.PCM = nn.Sequential(
                nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                nn.BatchNorm2d(mlp_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(dim, dim, 3, 1, 1, 1, group),
                nn.SiLU(inplace=True),
                )
        self.H = 0
        self.W = 0

    def forward(self, x):
        b, n, c = x.shape

        H = self.H
        W = self.W

        x_2d = x.view(b, H, W, c).permute(0, 3, 1, 2).contiguous()
        convX = self.drop_path(self.PCM(x_2d).permute(0, 2, 3, 1).contiguous().view(b, n, c))

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + convX

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x