"""This script prepares the model for training by freezing certain layers
"""

from loguru import logger
from models.mlp import MLP


def prepare_model_for_training(model, cfg, verbose=True):
    """
    Freeze certain layers of the model based on the configuration.
    
    Args:
        model (torch.nn.Module): The model to be prepared for training.
        cfg (dict): Configuration dictionary containing model settings.
    
    Returns:
        torch.nn.Module: The modified model with specific layers frozen.
    """
    # get training mode
    mode = cfg['MODE']
    if verbose:
        logger.warning(f'Preparing model for training in {mode} mode')
    
    if mode == 'vpt':
        for k, p in model.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
            if "head" in k:
                p.requires_grad = True
        
    elif mode == 'linear':
        # we freeze all layers except the head
        for k, p in model.named_parameters():
            if "head" not in k:
                p.requires_grad = False
            else:
                p.requires_grad = True    
    
    elif mode == 'partial_1':
        # we unfreeze only the last layer of the backbone (named as blocks.0 to blocks.11) and the head (so we have to unfreeze blocks.11)
        for k, p in model.named_parameters():
            if "blocks.11" in k or "head" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    elif mode == 'mlp_3':
        # we replace head with mlp with 3 layers
        model.head = MLP(
            input_dim=model.head.in_features,
            mlp_dims=[model.head.in_features]*3 + [model.head.out_features],
            special_bias=True
        ).cuda()
        for k, p in model.named_parameters():
            if "head" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    elif mode == 'bias':
        # unfreeze all bias terms of the model
        for k, p in model.named_parameters():
            if "bias" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
            if "head" in k:
                p.requires_grad = True
    
    elif mode == 'adapter':
        # unfreeze all adapter layers
        for k, p in model.named_parameters():
            if "adapter" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
            if "head" in k:
                p.requires_grad = True
    
    elif mode == 'side':
        # unfreeze all side layers and the head
        for k, p in model.named_parameters():
            if "side" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
            if "head" in k:
                p.requires_grad = True
            
    # display all parameters that will be trained
    if verbose:
        for k, p in model.named_parameters():
            if p.requires_grad:
                logger.warning(f'Unfrozen params: {k}')  
            
    
    # count the number of parameters that will be trained
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        logger.warning(f'Number of parameters to be trained: {num_params}')
    
    return model