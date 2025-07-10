"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""


from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn

import torchvision

import monai

class Conv3dBlock(nn.Module):
    """Conv3D Block to be used in Conv3dPatch class

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self,  
                 in_chans=1,
                 out_chans=128,
                 kernel_patch=(3,3,3),
                 stride=1,
                 padding=1,
                 p_drop = 0.):
        super().__init__()

        self.block = nn.Sequential(OrderedDict([
            ('conv3d', nn.Conv3d(in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_patch, stride=stride, padding=padding, bias=False)),
            ('bn3d', nn.BatchNorm3d(num_features=out_chans)),
            ('maxpool3d', nn.MaxPool3d(kernel_size=kernel_patch)),
            ('l_relu', nn.LeakyReLU(0.2)),
            ('dropout3d', nn.Dropout3d(p_drop))
        ]))
        

    def forward(self, x):
        return self.block(x)

class Conv3dPatch(nn.Module):
    """Convolutional Network to extract 3D patches

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self,  
                 in_chans=[1, 128, 192, 256, 512],
                 num_blocks=4,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 p_drop = 0.2):
        super().__init__()
        self.in_chans = in_chans
        self.num_blocks = in_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.p_drop = p_drop
        self.kernel_patch = tuple([self.kernel_size for _ in range(3)])        

        self.blocks = nn.ModuleList([
                Conv3dBlock(
                    in_chans=in_chans[ii],
                    out_chans=in_chans[ii+1],
                    kernel_patch=self.kernel_patch,
                    stride=self.stride,
                    padding=self.padding,
                    p_drop = self.p_drop
                )
                for ii in range(num_blocks)
            ]
        )
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x

class UNet3DBlock(nn.Module):
    """Conv3D Block to be used in Conv3dPatch class

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self,  
                 chans=[1, 32, 64],
                 kernel_patch=(2,2,2),
                 stride=1,
                 padding=1,
                 bias=False):
        super().__init__()

        self.block = nn.Sequential(OrderedDict([
            ('conv3d1', nn.Conv3d(in_channels=chans[0], out_channels=chans[1], kernel_size=kernel_patch, stride=stride, padding=padding, bias=False)),
            ('bn3d1', nn.BatchNorm3d(num_features=chans[1])),
            ('relu1', nn.ReLU()),
            ('conv3d2', nn.Conv3d(in_channels=chans[1], out_channels=chans[2], kernel_size=kernel_patch, stride=stride, padding=padding, bias=False)),
            ('bn3d2', nn.BatchNorm3d(num_features=chans[2])),
            ('relu2', nn.ReLU()),
            ('maxpool3d', nn.AvgPool3d(kernel_size=kernel_patch)),
        ]))
        

    def forward(self, x):
        return self.block(x)

class UNet3DEncoder(nn.Module):
    """Convolutional Network to extract 3D patches

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self,  
                 in_chans=[[1, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 256]],
                 num_blocks=4,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 bias=False):
        super().__init__()
        self.in_chans = in_chans
        self.num_blocks = len(in_chans)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_patch = tuple([self.kernel_size for _ in range(3)])
        self.bias = bias

        self.blocks = nn.ModuleList([
                UNet3DBlock(
                    chans=in_chans[ii],
                    kernel_patch=self.kernel_patch,
                    stride=self.stride,
                    padding=self.padding,
                    bias=bias
                )
                for ii in range(num_blocks)
            ]
        )
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        return x

class UNet3DEncoderClassifier(nn.Module):
    """Convolutional Network to extract 3D patches

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self,  
                 in_chans=[[1, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 256]],
                 num_blocks=4,
                 num_classes=2,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 bias=False):
        super().__init__()
        self.in_chans = in_chans
        self.num_blocks = len(in_chans)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_patch = tuple([self.kernel_size for _ in range(3)])
        self.bias = bias

        self.blocks = nn.ModuleList([
                UNet3DBlock(
                    chans=in_chans[ii],
                    kernel_patch=self.kernel_patch,
                    stride=self.stride,
                    padding=self.padding,
                    bias=bias
                )
                for ii in range(num_blocks)
            ]
        )
        
        sample_data = torch.rand((1,1,128,128,128))
        for blk in self.blocks:
            sample_data = blk(sample_data)
        
        self.head = nn.Linear(sample_data.view(1,-1).shape[1], num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        for blk in self.blocks:
            x = blk(x)
        
        x = x.view(B, -1)
        x = self.head(x)
        return x

class MyPEConv3d(nn.Module):
    """

    My patch extractor-based 3D convolutional network

    """
    def __init__(self,  
                 embed_dim=768,
                 num_classes=2,
                 kernel_size=5
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        self.proj = nn.Sequential(
                nn.Conv3d(1, self.embed_dim//4, self.kernel_size),
                nn.BatchNorm3d(self.embed_dim//4),
                nn.ReLU(),
                nn.AvgPool3d(3),
                nn.Conv3d(self.embed_dim//4, self.embed_dim//2, self.kernel_size),
                nn.BatchNorm3d(self.embed_dim//2),
                nn.ReLU(),
                nn.AvgPool3d(3),
                nn.Conv3d(self.embed_dim//2, self.embed_dim, self.kernel_size),
                nn.BatchNorm3d(self.embed_dim),
                nn.ReLU()
            )
        
        sample_data = torch.rand((1,1,128,128,128))
        sample_data = self.proj(sample_data)
        
        self.head = nn.Linear(sample_data.view(1,-1).shape[1], num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        x = self.proj(x)
        
        x = x.view(B, -1)
        x = self.head(x)
        
        return x

def make_resnet503d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    """Make 3D ResNet50 model from monai package

    Parameters
    ----------
    pretrained : bool, optional
        whether to use pre-trained weights or not, by default False
    spatial_dims : int, optional
        how many dims in a data sample, by default 3
    n_input_channels : int, optional
        first conv input channel number, by default 1
    num_classes : int, optional
        number of classes, by default 2
    """
    return monai.networks.nets.resnet50(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)

def make_resnet1013d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    """Make 3D ResNet50 model from monai package

    Parameters
    ----------
    pretrained : bool, optional
        whether to use pre-trained weights or not, by default False
    spatial_dims : int, optional
        how many dims in a data sample, by default 3
    n_input_channels : int, optional
        first conv input channel number, by default 1
    num_classes : int, optional
        number of classes, by default 2
    """
    return monai.networks.nets.resnet101(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)

def make_resnet1523d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    """Make 3D ResNet50 model from monai package

    Parameters
    ----------
    pretrained : bool, optional
        whether to use pre-trained weights or not, by default False
    spatial_dims : int, optional
        how many dims in a data sample, by default 3
    n_input_channels : int, optional
        first conv input channel number, by default 1
    num_classes : int, optional
        number of classes, by default 2
    """
    return monai.networks.nets.resnet152(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)

def make_densenet2013d(spatial_dims=3, 
                       n_input_channels=1, 
                       num_classes=2):
    """Make 3D DenseNet201 model from monai package

    Parameters
    ----------
    pretrained : bool, optional
        whether to use pre-trained weights or not, by default False
    spatial_dims : int, optional
        how many dims in a data sample, by default 3
    n_input_channels : int, optional
        first conv input channel number, by default 1
    num_classes : int, optional
        number of classes, by default 2
    """
    return monai.networks.nets.DenseNet201(spatial_dims=spatial_dims,
                                           in_channels=n_input_channels,
                                           out_channels=num_classes)


class ResNet50PE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet50PE, self).__init__()

        self.encoder = torchvision.models.resnet50()
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        
        self.conv1 = nn.Conv2d(2048, out_channels, 1)
    
    def forward(self, x):        
        out = self.encoder.conv1(x)
        out = self.encoder.bn1(out)
        out = self.encoder.relu(out)
        out = self.encoder.maxpool(out)
        out = self.encoder.layer1(out)
        out = self.encoder.layer2(out)
        out = self.encoder.layer3(out)
        out = self.encoder.layer4(out)
        
        out = self.conv1(out)
        
        return out

# Unet2D encoder

class UNet2DEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2DEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.maxpool(conv1_out))
        conv3_out = self.conv3(self.maxpool(conv2_out))
#         conv4_out = self.conv4(self.maxpool(conv3_out))
#         conv5_out = self.conv5(self.maxpool(conv4_out))
        return conv3_out

