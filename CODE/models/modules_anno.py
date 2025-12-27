import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from typing import Callable


class ClusterAttention(nn.Module):
    """
    Cluster-based channel attention used in CLAdapter.
    x: [B, N, C] -> out: [B, N, C]
    """
    def __init__(self, dim: int, len_token: int, centers: int,
                 temp_dim: int = 256, proj_drop: float = 0.0):
        super().__init__()
        self.len_token = len_token
        self.dim = dim
        self.centers = nn.Parameter(torch.randn(dim, centers))        # [C, K]
        self.tran_ms = nn.Parameter(torch.randn(centers, temp_dim, temp_dim))  # [K, C', C']
        self.down = nn.Linear(dim, temp_dim)   # C -> C'
        self.up = nn.Linear(temp_dim, dim)     # C' -> C
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        assert N == self.len_token and C == self.dim
        q = x.mean(dim=1)  # [B, C], global summary per sample

        # attention over K cluster centers in channel space
        attn = torch.mm(F.normalize(q, dim=-1),
                        F.normalize(self.centers, dim=0))  # [B, K]
        attn = attn.softmax(dim=-1)  # [B, K]

        x_down = self.down(x)        # [B, N, C']
        _, _, Cp = x_down.shape

        # build sample-specific channel transform: [B, C', C']
        attn_exp = attn.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, Cp, Cp)  # [B, K, C', C']
        tm = self.tran_ms.unsqueeze(0).expand_as(attn_exp)                # [B, K, C', C']
        tm = (attn_exp * tm).sum(dim=1)                                   # [B, C', C']

        x_out = torch.bmm(x_down, tm)  # [B, N, C']
        x_out = self.up(x_out)         # [B, N, C]
        return self.drop(x_out)


class FRResBlock(nn.Module):
    """ Feature refinement residual block used in CLAdapter. """
    def __init__(self, d_model: int, len_token: int, centers: int,
                 mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = ClusterAttention(d_model, len_token, centers)
        self.ln2 = LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(d_model, hidden)),
            ("act", act_layer()),
            ("proj", nn.Linear(hidden, d_model)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln1(x))            # cluster-based channel transform
        x = x + self.mlp(self.ln2(x))         # residual MLP refinement
        return x


class CLAdapter(nn.Module):
    """
    Stack of FRResBlocks that adapts frozen backbone features
    to data-limited scientific downstream tasks.
    """
    def __init__(self, check_point: list, width: int, len_token: int,
                 centers: int, depth: int, mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU):
        super().__init__()
        self.check_point = check_point  # [use_ckpt, num_ckpt_layers]
        self.blocks = nn.ModuleList([
            FRResBlock(width, len_token, centers, mlp_ratio, act_layer)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_ckpt, num_ckpt = self.check_point
        if use_ckpt and not torch.jit.is_scripting():
            for i, blk in enumerate(self.blocks):
                if i < num_ckpt:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        return x
