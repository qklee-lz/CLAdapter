import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from torch.utils.checkpoint import checkpoint

class Cluster_Attention(nn.Module):
    # * Token dim
    # def __init__(self, dim, len_token, centers, qkv_bias=False, proj_drop=0.):
    #     super().__init__()
    #     self.centers = nn.Parameter(torch.randn(dim, centers))
    #     self.tran_ms = nn.Parameter(torch.randn(centers, len_token, len_token))
    #     self.proj = nn.Linear(dim, dim)
    #     self.proj_drop = nn.Dropout(proj_drop)
    # def forward(self, x):
    #     B, N, C = x.shape
    #     q = x.mean(1)
    #     attn = torch.mm(F.normalize(q, dim=-1), F.normalize(self.centers, dim=0))
    #     attn = attn.softmax(dim=-1)
    #     attn = attn.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, N)
    #     tm = self.tran_ms.unsqueeze(0).expand_as(attn)
    #     tm = (attn*tm).sum(1)
    #     x = torch.bmm(tm, x)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    # * Channel dim
    def __init__(self, dim, len_token, centers, temp_dim=256, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(dim, centers))
        self.tran_ms = nn.Parameter(torch.randn(centers, temp_dim, temp_dim))
        self.down_layer = nn.Linear(dim, temp_dim)
        self.up_layer = nn.Linear(temp_dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        q = x.mean(1)
        x = self.down_layer(x)
        B, N, C = x.shape
        attn = torch.mm(F.normalize(q, dim=-1), F.normalize(self.centers, dim=0))
        attn = attn.softmax(dim=-1)
        attn = attn.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, C, C)
        tm = self.tran_ms.unsqueeze(0).expand_as(attn)
        tm = (attn*tm).sum(1)
        x = torch.bmm(x, tm)
        x = self.up_layer(x)
        x = self.proj_drop(x)
        return x

class FR_Resblock(nn.Module):
    def __init__(
            self,
            d_model: int,
            len_token: int,
            centers: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = Cluster_Attention(d_model, len_token, centers)
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def forward(self, x: torch.Tensor):
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CLAdapter(nn.Module):
    def __init__(self, check_point: list, width: int, len_token: int, centers: int, dt_layers: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.check_point = check_point
        self.da_resblocks = nn.ModuleList([FR_Resblock(width, len_token, centers, mlp_ratio, act_layer=act_layer) for i in range(dt_layers)])

    def forward(self, x: torch.Tensor):
        if self.check_point[0] and not torch.jit.is_scripting():
            for idx, dr in enumerate(self.da_resblocks):
                if idx<self.check_point[1]:
                    x = checkpoint(dr, x, use_reentrant=False)
                else:
                    x = dr(x)
        else:
            for dr in self.da_resblocks:
                x = dr(x)
        return x