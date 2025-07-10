"""

Patch Embedding functions for Vision Transformer in vit3d.py

Based primarily on a video tutorial from Vision Transformer

and 

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""



import torch
import torch.nn as nn

# from .resnetv2 import ResNetV2
from .convnets import ResNet50PE, UNet2DEncoder

class PatchEmbed2D(nn.Module):
    """
    Split image into patches and then embed them.

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
    def __init__(self, img_size, 
                 patch_size, 
                 embed_dim=768, 
                 patch_embed_fun='conv2d'):
        super().__init__()
        self.img_size = img_size
        # self.patch_size = patch_size
        self.n_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
        if patch_embed_fun == 'conv2d':
            self.proj = nn.Conv2d(
                in_channels=img_size[0],
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        elif patch_embed_fun == 'resnet502d':
            self.proj = ResNet50PE(img_size[0], embed_dim)
        elif patch_embed_fun == 'unet2d':
            self.proj = UNet2DEncoder(img_size[0], embed_dim)
        elif patch_embed_fun == 'mype2d':
            self.proj = nn.Sequential(
                nn.Conv2d(128, 192, 5),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.AvgPool2d(3),
                nn.Conv2d(192, 384, 5),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.AvgPool2d(3),
                nn.Conv2d(384, 768, 5),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )

        sample_torch = torch.rand(self.img_size)
        sample_torch = torch.unsqueeze(sample_torch, 0)
        out = self.proj(sample_torch)
        self.n_patches = out.flatten(2).shape[2]
        print('2D Patch Embedding Function Built.')

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x = self.proj(x).flatten(2) # out: (n_samples, embed_dim, n_patches[0], n_patches[1])
        # x = x.view(-1, self.e)
        # x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)

        return x

class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.

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
    def __init__(self, img_size, 
                 patch_size, 
                 embed_dim=768, 
                 patch_embed_fun='conv2d',
                 use_separation=True):
        super().__init__()
        self.img_size = img_size
        # self.patch_size = patch_size
        self.n_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.use_separation = use_separation
        
        if patch_embed_fun == 'conv2d':
            self.proj = nn.Conv2d(
                in_channels=img_size[0],
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        elif patch_embed_fun == 'resnet502d':
            self.proj = ResNet50PE(img_size[0], embed_dim)
        elif patch_embed_fun == 'unet2d':
            self.proj = UNet2DEncoder(img_size[0], embed_dim)
        elif patch_embed_fun == 'mype2d':
            self.proj = nn.Sequential(
                nn.Conv2d(128, 192, 5),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.AvgPool2d(3),
                nn.Conv2d(192, 384, 5),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.AvgPool2d(3),
                nn.Conv2d(384, 768, 5),
                nn.BatchNorm2d(768),
                nn.ReLU()
            )
        
        if self.use_separation:
            self.sep_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sep_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sep_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        sample_torch = torch.rand(self.img_size)
        sample_torch = torch.unsqueeze(sample_torch, 0)
        out = self.proj(sample_torch)
        out1 = self.proj(sample_torch.permute(0,2,3,1))
        out2 = self.proj(sample_torch.permute(0,3,2,1))
        if self.use_separation:
            self.n_patches = out.flatten(2).shape[2] + out1.flatten(2).shape[2] + out2.flatten(2).shape[2] + 3
        else:
            self.n_patches = out.flatten(2).shape[2] + out1.flatten(2).shape[2] + out2.flatten(2).shape[2]
        print('2D Patch Embedding Function Built.')

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x1 = self.proj(x).flatten(2) # out: (n_samples, embed_dim, n_patches[0], n_patches[1])
        x2 = self.proj(x.permute(0,2,3,1)).flatten(2) # out: (n_samples, embed_dim, n_patches[0] * n_patches[1])
        x3 = self.proj(x.permute(0,3,2,1)).flatten(2) # out: (n_samples, embed_dim, n_patches[0] * n_patches[1])
        if self.use_separation:
            sep_token1 = self.sep_token1.expand(x1.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            sep_token2 = self.sep_token2.expand(x2.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            sep_token3 = self.sep_token3.expand(x3.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            x = torch.cat((x1, sep_token1.transpose(1,2), x2, sep_token2.transpose(1,2), x3, sep_token3.transpose(1,2)), dim=2) # out: (n_samples, embed_dim, n_patches + 3)
        else:
            x = torch.cat((x1, x2, x3), dim=2)
        # x = x.view(-1, self.e)
        # x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches + 3, embed_dim)

        return x

class PatchEmbed3X(nn.Module):
    """
    Split image into patches and then embed them.
    3 different Conv2Ds for each orientation separately
    
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
    def __init__(self, img_size, patch_size, embed_dim=768, patch_embed_fun='3xconv2d', use_separation=True):
        super().__init__()
        self.img_size = img_size
        # self.patch_size = patch_size
        self.n_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.use_separation = use_separation
        
        # self.proj = nn.Conv2d(
        #     in_channels=img_size[0],
        #     out_channels=embed_dim,
        #     kernel_size=patch_size,
        #     stride=patch_size
        # )
        
        if patch_embed_fun == '3xconv2d':
            self.proj1 = nn.Conv2d(in_channels=img_size[0], out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
            self.proj2 = nn.Conv2d(in_channels=img_size[1], out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
            self.proj3 = nn.Conv2d(in_channels=img_size[2], out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_embed_fun == '3xmype2d':
            self.proj1 = nn.Sequential(nn.Conv2d(128, embed_dim//4, 5), nn.BatchNorm2d(embed_dim//4), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//4, embed_dim//2, 5), nn.BatchNorm2d(embed_dim//2), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//2, embed_dim, 5), nn.BatchNorm2d(embed_dim), nn.ReLU())
            self.proj2 = nn.Sequential(nn.Conv2d(128, embed_dim//4, 5), nn.BatchNorm2d(embed_dim//4), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//4, embed_dim//2, 5), nn.BatchNorm2d(embed_dim//2), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//2, embed_dim, 5), nn.BatchNorm2d(embed_dim), nn.ReLU())
            self.proj3 = nn.Sequential(nn.Conv2d(128, embed_dim//4, 5), nn.BatchNorm2d(embed_dim//4), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//4, embed_dim//2, 5), nn.BatchNorm2d(embed_dim//2), nn.ReLU(), nn.AvgPool2d(3),
                                      nn.Conv2d(embed_dim//2, embed_dim, 5), nn.BatchNorm2d(embed_dim), nn.ReLU())

        # self.proj = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[0])
        # self.proj1 = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[1])
        # self.proj2 = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[2])

        if self.use_separation:
            self.sep_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sep_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sep_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # proj
        sample_torch = torch.rand(self.img_size) # sagittal
        sample_torch = torch.unsqueeze(sample_torch, 0)
        out = self.proj1(sample_torch)
        # proj1
        out1 = self.proj2(sample_torch.permute(0,2,3,1))
        # proj2
        out2 = self.proj3(sample_torch.permute(0,3,2,1))
        
        if self.use_separation:
            self.n_patches = out.flatten(2).shape[2] + out1.flatten(2).shape[2] + out2.flatten(2).shape[2] + 3
        else:
            self.n_patches = out.flatten(2).shape[2] + out1.flatten(2).shape[2] + out2.flatten(2).shape[2]

        print('3x 2D Patch Embedding Function Built.')

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x1 = self.proj1(x).flatten(2) # out: (n_samples, embed_dim, n_patches[0], n_patches[1])
        x2 = self.proj2(x.permute(0,2,3,1)).flatten(2)
        x3 = self.proj3(x.permute(0,3,2,1)).flatten(2)
        if self.use_separation:
            sep_token1 = self.sep_token1.expand(x1.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            sep_token2 = self.sep_token2.expand(x2.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            sep_token3 = self.sep_token3.expand(x3.shape[0], -1, -1) # (n_samples, 1, embed_dim)
            x = torch.cat((x1, sep_token1.transpose(1,2), x2, sep_token2.transpose(1,2), x3, sep_token3.transpose(1,2)), dim=2) # out: (n_samples, embed_dim, n_patches + 3)
        else:
            x = torch.cat((x1, x2, x3), dim=2)
        # x = x.view(-1, self.e)
        # x = x # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)

        return x

class ProgressivePatchEmbed(nn.Module):
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
    def __init__(self, img_size, patch_size, embed_dim=768, patch_embed_fun='progressive_conv2d'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # sample random tensor to calculate the output shape
        sample_torch = torch.rand((1, *self.img_size)) # --> e.g. (1,1,128,128,128)
        
        self.patch_sizes = [128, 64, 32, 16]
        
        if patch_embed_fun == 'progressive_conv2d':
            
            self.proj = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=embed_dim,
                        kernel_size=self.patch_sizes[ii],
                        stride=self.patch_sizes[ii]
                    )
                    for ii in range(4)
                ]
            )
            
        xs = []
        
        for ii in range(0,4):
            xs.append(self.proj[ii](sample_torch[:, ii*16:(ii+1)*16, :, :]))
            

        for ii in range(4,8):
            xs.append(self.proj[abs(ii-7)](sample_torch[:, ii*16:(ii+1)*16, :, :]))

        out = xs[0].flatten(2)
        for jj in range(1, len(xs)):
            out = torch.cat((out, xs[jj].flatten(2)), dim=2)
        
        self.n_patches = out.shape[2]

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        # x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1], n_patches[2])
        # x = x.view(-1, self.e)
        # x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        
        xs = []
        
        for ii in range(0,4):
            xs.append(self.proj[ii](x[:, ii*16:(ii+1)*16, :, :]))

        for ii in range(4,8):
            xs.append(self.proj[abs(ii-7)](x[:, ii*16:(ii+1)*16, :, :]))

        out = xs[0].flatten(2)
        for jj in range(1, len(xs)):
            out = torch.cat((out, xs[jj].flatten(2)), dim=2)
        
        out = out.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)///

        return out

class ProgressivePatchEmbed3D(nn.Module):
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
    def __init__(self, img_size, patch_size, embed_dim=768, patch_embed_fun='progressive_conv3d'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # sample random tensor to calculate the output shape
        sample_torch = torch.rand((1, 1, *self.img_size)) # --> e.g. (1,1,128,128,128)
        
        self.patch_sizes = [128, 64, 32, 16]
        
        if patch_embed_fun == 'progressive_conv3d':
            
            self.proj = nn.ModuleList(
                [
                    nn.Conv3d(
                        in_channels=1,
                        out_channels=embed_dim,
                        kernel_size=(16, self.patch_sizes[ii], self.patch_sizes[ii]),
                        stride=(16, self.patch_sizes[ii], self.patch_sizes[ii])
                    )
                    for ii in range(4)
                ]
            )
            
        xs = []
        
        for ii in range(0,4):
            xs.append(self.proj[ii](sample_torch[:, :, ii*16:(ii+1)*16, :, :]))
            

        for ii in range(4,8):
            xs.append(self.proj[abs(ii-7)](sample_torch[:, :, ii*16:(ii+1)*16, :, :]))

        out = xs[0].flatten(2)
        for jj in range(1, len(xs)):
            out = torch.cat((out, xs[jj].flatten(2)), dim=2)
        
        self.n_patches = out.shape[2]

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        # x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1], n_patches[2])
        # x = x.view(-1, self.e)
        # x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        
        xs = []
        
        for ii in range(0,4):
            xs.append(self.proj[ii](x[:, :, ii*16:(ii+1)*16, :, :]))

        for ii in range(4,8):
            xs.append(self.proj[abs(ii-7)](x[:, :, ii*16:(ii+1)*16, :, :]))

        out = xs[0].flatten(2)
        for jj in range(1, len(xs)):
            out = torch.cat((out, xs[jj].flatten(2)), dim=2)
        
        out = out.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)///

        return out


# class PatchEmbed3X(nn.Module):
#     """
#     Split image into patches and then embed them.

#     Parameters
#     ----------
#     img_size : int (square)
#     patch_size : int (square)
#     in_chans : int
#     embed_dim : int

#     Atttributes:
#     -----------
#     n_patches : int
#     proj : nn.Conv2d

#     """
#     def __init__(self, img_size, patch_size, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         # self.patch_size = patch_size
#         self.n_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
#         # self.proj = nn.Conv2d(
#         #     in_channels=img_size[0],
#         #     out_channels=embed_dim,
#         #     kernel_size=patch_size,
#         #     stride=patch_size
#         # )

#         self.proj = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[0])
#         self.proj1 = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[1])
#         self.proj2 = ResNetV2(block_units = (3, 4, 9), width_factor = 1, in_chans = self.img_size[2])

#         # proj
#         sample_torch = torch.rand(self.img_size) # sagittal
#         sample_torch = torch.unsqueeze(sample_torch, 0)
#         out = self.proj(sample_torch)

#         # proj1
#         sample_torch = torch.rand(self.img_size).permute(1, 2, 0) # axial
#         sample_torch = torch.unsqueeze(sample_torch, 0)
#         out1 = self.proj1(sample_torch)

#         # proj2
#         sample_torch = torch.rand(self.img_size).permute(2, 1, 0) # coronal
#         sample_torch = torch.unsqueeze(sample_torch, 0)
#         out2 = self.proj2(sample_torch)
        
#         # self.n_patches = out.flatten(2).shape[2]
#         self.n_patches = out.flatten(2).shape[2] + out1.flatten(2).shape[2] + out2.flatten(2).shape[2]

#     def forward(self, x):
#         """
#         Input
#         ------
#         x : Shape (n_samples, in_chans, img_size, img_size)

#         Returns:
#         --------
#         Shape (n_samples, n_patches, embed_dims)
#         """
#         x0 = self.proj(x).flatten(2) # out: (n_samples, embed_dim, n_patches[0], n_patches[1])
#         x1 = self.proj1(x.permute(0,2,3,1)).flatten(2)
#         x2 = self.proj2(x.permute(0,3,2,1)).flatten(2)
#         x = torch.cat((x0, x1, x2), dim=2)
#         # x = x.view(-1, self.e)
#         # x = x # out: (n_samples, embed_dim, n_patches)
#         x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)

#         return x
