"""

3D ViT transformer that inputs 5D (n_batches, n_channels, height, weight, depth)

Based primarily on a video tutorial from Vision Transformer

and 

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""

import math
import numpy as np

import torch
import torch.nn as nn

# from .resnetv2 import ResNetV2

from utils.weight_init import trunc_normal_, init_weights_vit_timm, get_init_weights_vit, named_apply
from utils.utils import get_3d_sincos_pos_embed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed3D(nn.Module):
    """
    Split image into 3D patches and then embed them.

    Parameters
    ----------
    img_size : int (square)
    patch_size : int (square)
    in_chans : int
    embed_dim : int

    Atttributes:
    -----------
    n_patches : int
    proj : nn.Conv2d

    """
    def __init__(self, img_size, patch_size, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        # sample random tensor to calculate the output shape
        sample_torch = torch.rand((1, 1, *self.img_size)) # --> e.g. (1,1,128,128,128)

        self.proj = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        out = self.proj(sample_torch)
        self.n_patches = out.flatten(2).shape[2]

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1], n_patches[2])
        # x = x.view(-1, self.e)
        x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)///

        return x

class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    -----------
    dim : int (dim per token features)
    n_heads : int
    qkv_bias : bool
    attn_p : float (Dropout applied to q, k, v)
    proj_p : float (Dropout applied to output tensor)

    Attributes
    ----------
    scale : float
    qkv : nn.Linear
    proj : nn.Linear
    attn_drop, proj_drop : nn.Dropout
    
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, n_patches + 1, dim)

        Returns:
        -------
        Shape (n_samples, n_patches + 1, dim)

        """
        n_samples, n_tokens, dim =  x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # each with (n_samples, n_heads, n_patches + 1, head_dim)

        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ----------
    in_features : int
    hidden_features : int
    out_features : int
    p : float

    Attributes
    ---------
    fc1 : nn.Linear
    act : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, in_features)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(
            x
            ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)

        return x

class Block(nn.Module):
    """
    Transformer block

    Parameters
    ----------
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes
    ----------
    norm1, norm2 : LayerNorm
    attn : Attention
    mlp : MLP
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, dim)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VPT_ViT3D(nn.Module):
    """
    3D Vision Transformer for Visual Prompt Tuning

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
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
                use_vpt=True,
                num_prompt_tokens=50,
                prompt_drop_rate=0.1,
                vpt_deep=False,
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

        trunc_normal_(self.cls_token, std=.02)

        self.init_weights(weight_init)
        
        self.use_vpt = use_vpt
        self.vpt_deep = vpt_deep
        print('USE VPT: ', self.use_vpt)
        if use_vpt:
            self.num_prompt_tokens = num_prompt_tokens
            self.prompt_dropout = nn.Dropout(p=prompt_drop_rate)
            self.prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()
            
            print(f"Prompt learning with {self.num_prompt_tokens} tokens and dropout {prompt_drop_rate}")
            
            # random initialization of prompt embeddings
            from functools import reduce
            from operator import mul
            import math
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size, patch_size), 1) + self.prompt_dim))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_prompt_tokens, self.prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            
            if vpt_deep:
                total_d_layer = depth - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_prompt_tokens, self.prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
    
    def incorporate_prompt(self, x):
        # combine prompt embeddings with input
        B = x.shape[0]
        # after CLS token, before patch tokens
        x = torch.cat((x[:, :1, :], self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)), x[:, 1:, :]), dim=1)
        # (batch_size, cls_token + prompt tokens + image tokens, embed_dim)
        
        return x
    
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
    
    # def train(self, mode=True):
    #     # set train status for this class: disable all but the prompt-related modules
    #     if mode:
    #         # training:
    #         for module in self.children():
    #             module.eval()
    #         self.prompt_proj.train()
    #         self.prompt_dropout.train()
    #     else:
    #         # eval:
    #         for module in self.children():
    #             module.train(mode)
    
    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = len(self.blocks)
        
        for ii in range(num_layers):
            if ii == 0:
                hidden_states = self.blocks[ii](embedding_output)
            else:
                if ii <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[ii-1].expand(B, -1, -1)))
                    
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :], 
                        deep_prompt_emb, 
                        hidden_states[:, (1+self.num_prompt_tokens):, :]
                        ), dim=1)
                    
                hidden_states = self.blocks[ii](hidden_states)
            
        return hidden_states
                    
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
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(
                n_samples, -1, -1
            ) # (n_samples, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        
        # incorporate prompt tokens
        x = self.incorporate_prompt(x)
        
        if self.vpt_deep:
            x = self.forward_deep_prompt(x)
        else:
            for block in self.blocks:
                x = block(x)
        
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0] if self.cls_token is not None else x.mean(dim=1)
        # cls_token_final = self.bottleneck(cls_token_final)
        x = self.head(cls_token_final)

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