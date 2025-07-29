"""
ViT3D SimCLR Model Definition
"""
import torch.nn as nn
from .vit3d import Vision_Transformer3D

class ViT3DSimCLR(nn.Module):
    """
    Vision Transformer 3D model for SimCLR training.
    Inherits from Vision_Transformer3D and sets the model for SimCLR.
    """
    def __init__(self, out_dim=128, **kwargs):
        super(ViT3DSimCLR, self).__init__()
        self.backbone = Vision_Transformer3D(**kwargs, 
                                             n_classes=out_dim)  # Initialize the ViT3D model with provided arguments
        dim_mlp = self.backbone.head.in_features
        
        self.backbone.head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.backbone.head
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.backbone(x)
        
    
    