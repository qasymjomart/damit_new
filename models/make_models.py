# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

import monai

# from make_dataloaders import make_dataloaders
from .vit3d import Vision_Transformer3D
from .vpt_vit3d import VPT_ViT3D
from .adapter_vit3d import Adapter_ViT3D
from .side_vit3d import SideTune_ViT3D
from .maskedautoencoder3d import MaskedAutoencoderViT3D
from .convnets import make_resnet503d, make_resnet1013d, make_resnet1523d, make_densenet2013d, UNet3DEncoderClassifier, MyPEConv3d

# from perceiver_pytorch import Perceiver
from .perceiver import Perceiver


_models_factory = {
    'ViT3D': Vision_Transformer3D,
    'ViT3D_monai': monai.networks.nets.ViT,
    'VPT': VPT_ViT3D,
    'AdapterViT3D': Adapter_ViT3D,
    'SideTune_ViT3D': SideTune_ViT3D,
    'ResNet50': make_resnet503d,
    'ResNet101': make_resnet1013d,
    'ResNet152': make_resnet1523d,
    'DenseNet201': make_densenet2013d,
    'UNet3DClassifier': UNet3DEncoderClassifier,
    'MyPEConv3d': MyPEConv3d,
    'MaskedAutoencoderViT3D': MaskedAutoencoderViT3D,
    'Perceiver': Perceiver
}

def make_model(cfg, args):
    """
    Make vanilla training mode Vision Transformer

    """
    
    if args.vit_size in ['small']:
        cfg['MODEL']['embed_dim'] = 384
        cfg['MODEL']['depth'] = 12
        cfg['MODEL']['n_heads'] = 6
    elif args.vit_size in ['large']:
        cfg['MODEL']['embed_dim'] = 1024
        cfg['MODEL']['depth'] = 24
        cfg['MODEL']['n_heads'] = 16
    elif args.vit_size in ['huge']:
        cfg['MODEL']['embed_dim'] = 1280
        cfg['MODEL']['depth'] = 32
        cfg['MODEL']['n_heads'] = 16
    
    if cfg['MODEL']['TYPE'] in ['ViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'],
            in_chans          = cfg['MODEL']['in_chans'],
            n_classes         = cfg['MODEL']['n_classes'],
            embed_dim         = cfg['MODEL']['embed_dim'],
            depth             = cfg['MODEL']['depth'],
            n_heads           = cfg['MODEL']['n_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            p                 = cfg['MODEL']['p'],
            attn_p            = cfg['MODEL']['attn_p'],
            global_avg_pool   = cfg['MODEL']['global_avg_pool'],
            pos_embed_type    = cfg['MODEL']['pos_embed_type'],
        )
        
    elif cfg['MODEL']['TYPE'] in ['VPT']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), f"{cfg['MODEL']['TYPE']} not in the model factory list"
        
        model = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'],
            in_chans          = cfg['MODEL']['in_chans'],
            n_classes         = cfg['MODEL']['n_classes'],
            embed_dim         = cfg['MODEL']['embed_dim'],
            depth             = cfg['MODEL']['depth'],
            n_heads           = cfg['MODEL']['n_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            p                 = cfg['MODEL']['p'],
            attn_p            = cfg['MODEL']['attn_p'],
            global_avg_pool   = cfg['MODEL']['global_avg_pool'],
            patch_embed_fun   = cfg['MODEL']['patch_embed_fun'],
            pos_embed_type    = cfg['MODEL']['pos_embed_type'],
            use_vpt           = cfg['VPT']['use_vpt'], 
            num_prompt_tokens = cfg['VPT']['num_prompt_tokens'],
            prompt_drop_rate  = cfg['VPT']['prompt_drop_rate'],
            vpt_deep          = cfg['VPT']['vpt_deep'],
        )
    
    elif cfg['MODEL']['TYPE'] in ['AdapterViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'],
            in_chans          = cfg['MODEL']['in_chans'],
            n_classes         = cfg['MODEL']['n_classes'],
            embed_dim         = cfg['MODEL']['embed_dim'],
            depth             = cfg['MODEL']['depth'],
            n_heads           = cfg['MODEL']['n_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            p                 = cfg['MODEL']['p'],
            attn_p            = cfg['MODEL']['attn_p'],
            global_avg_pool   = cfg['MODEL']['global_avg_pool'],
            pos_embed_type    = cfg['MODEL']['pos_embed_type'],
        )
        
        print('AdapterViT3D model built.')
    
    elif cfg['MODEL']['TYPE'] in ['SideTune_ViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'],
            in_chans          = cfg['MODEL']['in_chans'],
            n_classes         = cfg['MODEL']['n_classes'],
            embed_dim         = cfg['MODEL']['embed_dim'],
            depth             = cfg['MODEL']['depth'],
            n_heads           = cfg['MODEL']['n_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            p                 = cfg['MODEL']['p'],
            attn_p            = cfg['MODEL']['attn_p'],
            global_avg_pool   = cfg['MODEL']['global_avg_pool'],
            pos_embed_type    = cfg['MODEL']['pos_embed_type'],
        )

        print('SideTune_ViT3D model built.')


    elif cfg['MODEL']['TYPE'] in ['ViT3D_monai']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            in_channels     = cfg['MODEL']['in_chans'],
            img_size        = cfg['MODEL']['img_size'],
            patch_size      = cfg['MODEL']['patch_size'],
            hidden_size     = cfg['MODEL']['embed_dim'],
            mlp_dim         = 4 * cfg['MODEL']['embed_dim'],
            num_layers      = cfg['MODEL']['depth'],
            num_heads       = cfg['MODEL']['n_heads'],
            pos_embed       = 'conv',
            classification  = True,
            num_classes     = cfg['MODEL']['n_classes'],
            dropout_rate    = cfg['MODEL']['p'],
            spatial_dims    = 3,
            post_activation = 'Tanh',
            qkv_bias        = cfg['MODEL']['qkv_bias']
        )

        print('ViT3D_monai model built.')

    elif cfg['MODEL']['TYPE'] in ['ResNet50', 'ResNet101', 'ResNet152', 'DenseNet201']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            spatial_dims     = 3,
            n_input_channels = 1,
            num_classes      = cfg['MODEL']['n_classes']
        )

        print(cfg['MODEL']['TYPE'], ' model built.')
    
    elif cfg['MODEL']['TYPE'] in ['UNet3DClassifier']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            in_chans    = [[1, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 256]],
            num_blocks  = 4,
            num_classes = cfg['MODEL']['n_classes'],
            kernel_size = 2,
            stride      = 1,
            padding     = 0,
            bias        = False
        )

        print('UNet3DClassifier model built.')
        
    elif cfg['MODEL']['TYPE'] in ['MyPEConv3d']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            embed_dim   = cfg['MODEL']['embed_dim'],
            num_classes = cfg['MODEL']['n_classes'],
            kernel_size = cfg['MODEL']['kernel_size']
        )

        print('MyPEConv3 model built.')
    
    elif cfg['MODEL']['TYPE'] in ['Perceiver']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = Perceiver(
            input_channels = 1,          # number of channels for each token of the input
            input_axis = 3,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                        #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            num_classes = cfg['MODEL']['n_classes'],          # output number of classes
            attn_dropout = cfg['MODEL']['attn_p'],
            ff_dropout = cfg['MODEL']['p'],
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 1,      # number of self attention blocks per cross attention
            patch_size = cfg['MODEL']['patch_size']
        )

        print('Perceiver model built.')
        print('Patch size was set to ', cfg['MODEL']['patch_size'])

    return model

def make_mae_model(cfg, args):
    """Build a 3D MAE (ViT-B based)
    to be used for pre-training

    """

    if cfg['MODEL']['TYPE'] in ['MaskedAutoencoderViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'], 
            in_chans          = cfg['MODEL']['in_chans'],
            embed_dim         = cfg['MODEL']['embed_dim'], 
            depth             = cfg['MODEL']['depth'], 
            num_heads         = cfg['MODEL']['n_heads'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            decoder_embed_dim = cfg['MODEL']['decoder_embed_dim'], 
            decoder_depth     = cfg['MODEL']['decoder_depth'], 
            decoder_num_heads = cfg['MODEL']['decoder_num_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'], 
            norm_pix_loss     = cfg['MODEL']['norm_pix_loss'],
            patch_embed_fun   = 'conv3d'
        )

        print('MAE ', cfg['MODEL']['TYPE'], ' model built.')
    
    elif cfg['MODEL']['TYPE'] in ['MaskedAutoencoderViT3DConvit']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'], 
            in_chans          = cfg['MODEL']['in_chans'],
            embed_dim         = cfg['MODEL']['embed_dim'], 
            depth             = cfg['MODEL']['depth'], 
            num_heads         = cfg['MODEL']['n_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'], 
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            qk_scale          = cfg['MODEL']['qk_scale'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            decoder_embed_dim = cfg['MODEL']['decoder_embed_dim'], 
            decoder_depth     = cfg['MODEL']['decoder_depth'], 
            decoder_num_heads = cfg['MODEL']['decoder_num_heads'],
            norm_pix_loss     = cfg['MODEL']['norm_pix_loss'],
            locality_strength = cfg['MODEL']['locality_strength'],
            local_up_to_layer = cfg['MODEL']['local_up_to_layer']
        )

        print('MAE ', cfg['MODEL']['TYPE'], ' model built.')

    return model_mae
