import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from torch.utils.checkpoint import checkpoint


class Cluster_Attention(nn.Module):
    """
    Cluster-based attention module with two modes:
        - mode="channel": cluster & transform along channel dimension
        - mode="token":   cluster & transform along token dimension

    Args:
        dim:        Channel dimension of input tokens (C).
        len_token:  Number of tokens (N) in the sequence.
        centers:    Number of cluster centers (K).
        temp_dim:   Reduced channel dimension C' for channel-mode.
        mode:       "channel" or "token".
        qkv_bias:   Reserved for compatibility (unused).
        proj_drop:  Dropout ratio on output.
    """

    def __init__(
        self,
        dim: int,
        len_token: int,
        centers: int,
        temp_dim: int = 256,
        mode: str = "channel",
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert mode in ["channel", "token"], "mode must be 'channel' or 'token'"
        self.mode = mode
        self.len_token = len_token
        self.dim = dim
        self.centers_num = centers

        # Learnable cluster centers in channel space: [C, K]
        self.centers = nn.Parameter(torch.randn(dim, centers))

        if self.mode == "channel":
            # Channel transform matrices for each center: [K, C', C']
            self.tran_ms = nn.Parameter(torch.randn(centers, temp_dim, temp_dim))

            # Channel reduction & expansion: C -> C' -> C
            self.down_layer = nn.Linear(dim, temp_dim)
            self.up_layer = nn.Linear(temp_dim, dim)
            self.temp_dim = temp_dim
        else:
            # Token transform matrices for each center: [K, N, N]
            # N = len_token (fixed for this module)
            self.tran_ms = nn.Parameter(torch.randn(centers, len_token, len_token))

        # Final dropout (shared for both modes)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
        Returns:
            out: [B, N, C]
        """
        B, N, C = x.shape
        assert N == self.len_token, f"len_token mismatch: expected {self.len_token}, got {N}"
        assert C == self.dim, f"channel dim mismatch: expected {self.dim}, got {C}"

        # Global query per sample by averaging over tokens: [B, C]
        q = x.mean(dim=1)  # [B, C]

        # Compute attention over cluster centers: [B, K]
        # Normalize along channel for both q and centers
        q_norm = F.normalize(q, dim=-1)                  # [B, C]
        centers_norm = F.normalize(self.centers, dim=0)  # [C, K]
        attn = torch.mm(q_norm, centers_norm)            # [B, K]
        attn = attn.softmax(dim=-1)                      # [B, K]

        if self.mode == "channel":
            # -------- Channel-mode: transform along channel dimension --------
            # 1) Reduce channel dim: [B, N, C] -> [B, N, C']
            x_down = self.down_layer(x)   # [B, N, C']
            _, _, Cp = x_down.shape

            # 2) Build sample-specific transform matrix:
            #    - per-center matrices: [K, C', C']
            #    - attention weights:   [B, K]
            #    -> expand to [B, K, C', C'], then weighted sum over K
            attn_expanded = attn.unsqueeze(-1).unsqueeze(-1)   # [B, K, 1, 1]
            attn_expanded = attn_expanded.repeat(1, 1, Cp, Cp) # [B, K, C', C']

            tm = self.tran_ms.unsqueeze(0).expand_as(attn_expanded)  # [B, K, C', C']
            tm = (attn_expanded * tm).sum(dim=1)                     # [B, C', C']

            # 3) Apply transform on channel dimension:
            #    [B, N, C'] x [B, C', C'] -> [B, N, C']
            x_trans = torch.bmm(x_down, tm)  # [B, N, C']

            # 4) Expand back to original channel dim: [B, N, C]
            x_out = self.up_layer(x_trans)

        else:
            # -------- Token-mode: transform along token dimension --------
            # 1) Per-center token transform: [K, N, N]
            # 2) Attention weights:          [B, K]
            # -> expand to [B, K, N, N], weighted sum over K
            attn_expanded = attn.unsqueeze(-1).unsqueeze(-1)     # [B, K, 1, 1]
            attn_expanded = attn_expanded.repeat(1, 1, N, N)     # [B, K, N, N]

            tm = self.tran_ms.unsqueeze(0).expand_as(attn_expanded)  # [B, K, N, N]
            tm = (attn_expanded * tm).sum(dim=1)                     # [B, N, N]

            # 3) Apply transform along token dimension:
            #    [B, N, N] x [B, N, C] -> [B, N, C]
            x_out = torch.bmm(tm, x)  # [B, N, C]

        # 5) Final dropout
        x_out = self.drop(x_out)
        return x_out


class FR_Resblock(nn.Module):
    """
    Feature Refinement Residual Block used inside CLAdapter.

    Structure:
        x -> LN -> Cluster_Attention -> x1
        x1 -> LN -> MLP -> x2
        out = x1 + x2
    """

    def __init__(
        self,
        d_model: int,
        len_token: int,
        centers: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        attn_mode: str = "channel",
        temp_dim: int = 256,
    ):
        """
        Args:
            d_model:    Channel dimension of tokens (C).
            len_token:  Number of tokens (N).
            centers:    Number of cluster centers (K).
            mlp_ratio:  Expansion ratio for MLP hidden width.
            act_layer:  Activation function in MLP.
            attn_mode:  'channel' or 'token' for Cluster_Attention.
            temp_dim:   Reduced channel dim for channel-mode.
        """
        super().__init__()

        # First LN + Cluster_Attention
        self.ln_1 = LayerNorm(d_model)
        self.attn = Cluster_Attention(
            dim=d_model,
            len_token=len_token,
            centers=centers,
            temp_dim=temp_dim,
            mode=attn_mode,
        )

        # Second LN + MLP
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("act", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
        Returns:
            out: [B, N, C]
        """
        # Cluster-based attention refinement (no residual here, following your original design)
        x = self.attn(self.ln_1(x))  # [B, N, C]

        # MLP with residual: out = x + MLP(LN(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLAdapter(nn.Module):
    """
    CLAdapter: stack of FR_Resblocks to adapt pre-trained backbone features
    to diverse scientific downstream tasks.

    Args:
        check_point: [flag, num_layers]
                     - flag:       whether to use gradient checkpointing.
                     - num_layers: how many early blocks use checkpointing.
        width:       Token channel dimension (C).
        len_token:   Number of tokens (N).
        centers:     Number of cluster centers (K) in Cluster_Attention.
        dt_layers:   Number of stacked FR_Resblocks (adapter depth).
        mlp_ratio:   Expansion ratio in MLP.
        act_layer:   Activation function for MLP.
        attn_mode:   "channel" or "token" (passes into Cluster_Attention).
        temp_dim:    Reduced channel dim for channel-mode.
    """

    def __init__(
        self,
        check_point: list,
        width: int,
        len_token: int,
        centers: int,
        dt_layers: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        attn_mode: str = "channel",
        temp_dim: int = 256,
    ):
        super().__init__()
        self.check_point = check_point  # e.g. [True, 2]
        self.width = width
        self.len_token = len_token
        self.centers = centers
        self.dt_layers = dt_layers
        self.attn_mode = attn_mode
        self.temp_dim = temp_dim

        # Build a stack of FR_Resblocks as the adapter
        self.da_resblocks = nn.ModuleList([
            FR_Resblock(
                d_model=width,
                len_token=len_token,
                centers=centers,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                attn_mode=attn_mode,
                temp_dim=temp_dim,
            )
            for _ in range(dt_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] features from a backbone (e.g., ViT tokens).
        Returns:
            out: [B, N, C] adapted features.
        """
        use_ckpt = self.check_point[0]
        num_ckpt_layers = self.check_point[1] if len(self.check_point) > 1 else 0

        # Use checkpointing only in eager mode (not in TorchScript)
        if use_ckpt and not torch.jit.is_scripting():
            for idx, block in enumerate(self.da_resblocks):
                if idx < num_ckpt_layers:
                    # Checkpoint early layers to save memory
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        else:
            # Standard forward without checkpointing
            for block in self.da_resblocks:
                x = block(x)

        return x


if __name__ == '__main__':
    # channel dim
    adapter = CLAdapter(
        check_point=[True, 2],
        width=768,
        len_token=196,
        centers=8,
        dt_layers=4,
        attn_mode="channel",
        temp_dim=256,
    )

    # token dim
    adapter_token = CLAdapter(
        check_point=[False, 0],
        width=768,
        len_token=196,
        centers=8,
        dt_layers=2,
        attn_mode="token",
    )
