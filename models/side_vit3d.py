#!/usr/bin/env python3
'''
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Based on also: https://github.com/KMnP/vpt/blob/main/src/models/vit_models.py
'''
import math
import numpy as np
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit3d import Block, PatchEmbed3D
from .convnets import make_resnet1523d

from utils.weight_init import trunc_normal_, init_weights_vit_timm, get_init_weights_vit, named_apply
from utils.utils import get_3d_sincos_pos_embed

class SideTune_ViT3D(nn.Module):
    def __init__(self, 
                img_size=384, 
                patch_size=16, 
                in_chans=3, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.,
                weight_init='',
                global_avg_pool=False,
                pos_embed_type='learnable',
                use_separation=True,
                **kwargs
                ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) if global_avg_pool == False else None
        embed_len = self.patch_embed.n_patches if global_avg_pool else 1 + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=True
            )
        
        if pos_embed_type == 'abs':
            self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=False
            )
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(np.cbrt(self.patch_embed.n_patches)), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            print('Abs pos embed built.')
            
        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
        self.side = make_resnet1523d(spatial_dims=len(img_size), n_input_channels=in_chans, num_classes=n_classes)
        self.side.fc = nn.Linear(self.side.fc.in_features, embed_dim, bias=False)
        # self.side.fc = nn.Identity()
        self.side_alpha = nn.Parameter(torch.tensor(0.8))
        
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights_vit_timm)
        self.init_weights(weight_init)
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights_vit_timm(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        if self.side is not None:
            side_x = self.side(x)
        
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(
                n_samples, -1, -1
            ) # (n_samples, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # just the CLS token
        x = x[:, 0] if self.cls_token is not None else x.mean(dim=1)
        
        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_x

        # cls_token_final = self.bottleneck(cls_token_final)
        x = self.head(x)

        return x
    
    def save(self, optimizer, scaler, checkpoint):
        state = {"net": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(state, checkpoint)

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()