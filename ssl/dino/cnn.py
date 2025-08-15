import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv3DBlock(nn.Module):
    """Basic 3D conv block with BatchNorm and ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleConv3DEncoder(nn.Module):
    """
    Simple 3D ConvNet Encoder that outputs 2048-dim features
    
    Architecture:
    Input: (B, 1, 128, 128, 128)
    Conv1: 64 channels, stride=2  -> (B, 64, 64, 64, 64)
    Conv2: 128 channels, stride=2 -> (B, 128, 32, 32, 32) 
    Conv3: 256 channels, stride=2 -> (B, 256, 16, 16, 16)
    Conv4: 512 channels, stride=2 -> (B, 512, 8, 8, 8)
    Conv5: 512 channels, stride=1 -> (B, 512, 8, 8, 8)
    GlobalAvgPool -> (B, 512)
    FC -> (B, 2048)
    """
    
    def __init__(self, in_channels: int = 1, **kwargs):
        super().__init__()
        
        # Progressive downsampling
        self.conv1 = SimpleConv3DBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = SimpleConv3DBlock(64, 128, stride=2)
        self.conv3 = SimpleConv3DBlock(128, 256, stride=2) 
        self.conv4 = SimpleConv3DBlock(256, 512, stride=2)
        self.conv5 = SimpleConv3DBlock(512, 512, stride=1)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(512, 2048)
        self.embed_dim = 2048
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 128, 128, 128)
            
        Returns:
            Feature tensor (B, 2048)
        """
        x = self.conv1(x)  # (B, 64, 64, 64, 64)
        x = self.conv2(x)  # (B, 128, 32, 32, 32)
        x = self.conv3(x)  # (B, 256, 16, 16, 16)
        x = self.conv4(x)  # (B, 512, 8, 8, 8)
        x = self.conv5(x)  # (B, 512, 8, 8, 8)
        
        x = self.global_pool(x)  # (B, 512, 1, 1, 1)
        x = x.flatten(1)  # (B, 512)
        x = self.fc1(x)  # (B, 2048)
        
        return x

if __name__ == "__main__":
    # Example usage
    model = SimpleConv3DEncoder()
    input_tensor = torch.randn(8, 1, 128, 128, 128)  # Batch of 8 samples
    output_features = model(input_tensor)
    print(output_features.shape)  # Should print: torch.Size([8, 2048])