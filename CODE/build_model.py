import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from utils import *
import timm
import open_clip

def build_FR(config):
    CLAdapter_Module = CLAdapter(check_point = config.MODEL.CLAdapter.checkpoint, 
                    width = config.MODEL.backbone.out_dim, 
                    len_token = config.MODEL.backbone.num_patch, 
                    centers = config.MODEL.CLAdapter.centers, 
                    dt_layers = config.MODEL.CLAdapter.layers,
                    mlp_ratio = config.MODEL.CLAdapter.mlp_ratio)
    return CLAdapter_Module

class Post_vit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.MODEL.backbone.out_dim)
        self.norm2 = nn.LayerNorm(config.MODEL.backbone.out_dim)
        self.CLAdapter_layers = build_FR(config)

    def forward(self, x):
        x = self.norm1(x)
        x = self.CLAdapter_layers(x)
        x = self.norm2(x)
        return x

class CLAdapter_CLIP_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.m_mode = config.MODEL.m_mode
        self.f_mode = config.MODEL.f_mode
        if ' ' not in config.MODEL.backbone.model_name:
            self.backbone = timm.create_model(config.MODEL.backbone.model_name, pretrained=True)
        else:
            names = config.MODEL.backbone.model_name.split()
            self.backbone = open_clip.create_model(names[0], names[1]).visual.trunk

        if self.f_mode != 'full' and config.MODEL.finetune is None:
            for param in self.backbone.parameters():
                param.requires_grad=False
        if self.f_mode == 'cla':
            self.post = Post_vit(config)
        self.head = nn.Linear(config.MODEL.backbone.out_dim, config.MODEL.num_classes)

        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()

    def forward(self, x, test=False):
        x = self.backbone.forward_features(x)
        if self.m_mode == 'conv' or self.m_mode == 'res_xcep':
            x = x.flatten(2).permute(0, 2, 1)
            if self.f_mode == 'cla':
                x = self.post(x)
            x = x.mean(1)
        else:
            if self.f_mode == 'cla':
                x = self.post(x[:, 1:, :]).mean(1)
            else:
                x = x[:, 0, :]
        x = self.head(x)
        return x


