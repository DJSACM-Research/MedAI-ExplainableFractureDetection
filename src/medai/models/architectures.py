"""
Neural Network Architectures for MedAI Fracture Detection.

Custom architectures including HyperColumn with CBAM attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "HyperColumnCBAMDenseNet169",
]


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for CBAM.
    
    Applies attention across channels using both average and max pooling
    to capture different aspects of the feature maps.
    
    Args:
        in_channels: Number of input channels.
        ratio: Reduction ratio for the bottleneck layer.
    """
    
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for CBAM.
    
    Applies attention across spatial dimensions using channel-wise
    average and max pooling.
    
    Args:
        kernel_size: Convolution kernel size (must be 3 or 7).
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(x_concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines Channel Attention and Spatial Attention sequentially
    to produce refined feature maps.
    
    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    
    Args:
        in_channels: Number of input channels.
        ratio: Reduction ratio for channel attention.
        kernel_size: Kernel size for spatial attention.
    """
    
    def __init__(self, in_channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel Attention
        x_ca = x * self.ca(x)
        # 2. Spatial Attention
        x_sa = x_ca * self.sa(x_ca)
        return x_sa


class HyperColumnCBAMDenseNet169(nn.Module):
    """
    HyperColumn DenseNet169 with CBAM Attention.
    
    Combines multi-scale features from DenseNet169 using HyperColumns,
    with CBAM attention for improved feature refinement.
    
    Architecture:
        1. DenseNet169 backbone for hierarchical feature extraction
        2. HyperColumn fusion of features from multiple dense blocks
        3. CBAM attention on fused features
        4. Global pooling and classification head
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout probability before classifier.
    
    Examples:
        >>> model = HyperColumnCBAMDenseNet169(num_classes=8)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> out = model(x)  # Shape: (1, 8)
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Load pre-trained DenseNet169 backbone
        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet169(weights=weights)

        # Extract feature extractor components
        self.features = densenet.features

        # Define sequential blocks for forward pass
        self.init_conv = nn.Sequential(
            self.features.conv0,
            self.features.norm0,
            self.features.relu0,
            self.features.pool0,
        )
        self.db1 = self.features.denseblock1
        self.t1 = self.features.transition1  # Output: 128 channels
        self.db2 = self.features.denseblock2
        self.t2 = self.features.transition2  # Output: 256 channels
        self.db3 = self.features.denseblock3
        self.t3 = self.features.transition3  # Output: 640 channels
        self.db4 = self.features.denseblock4
        self.norm_final = self.features.norm5  # Output: 1664 channels

        # Calculate HyperColumn fusion channels
        # Total: 1664 + 640 + 256 + 128 = 2688
        hc_channels = (
            self.norm_final.num_features  # 1664
            + self.t3.conv.out_channels   # 640
            + self.t2.conv.out_channels   # 256
            + self.t1.conv.out_channels   # 128
        )

        # Fusion layer to reduce channel count
        self.fusion_conv = nn.Conv2d(hc_channels, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)

        # CBAM attention on fused features
        self.cbam = CBAM(1024)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Initial convolution layers
        x = self.init_conv(x)

        # 2. Dense blocks with transition layers (HyperColumn sources)
        x = self.db1(x)
        t1_out = self.t1(x)

        x = self.db2(t1_out)
        t2_out = self.t2(x)

        x = self.db3(t2_out)
        t3_out = self.t3(x)

        x = self.db4(t3_out)
        x_final = self.norm_final(x)

        # 3. HyperColumn Fusion
        target_size = x_final.shape[2:]

        # Upsample intermediate features to final feature map size
        t1_resized = F.interpolate(
            t1_out, size=target_size, mode="bilinear", align_corners=False
        )
        t2_resized = F.interpolate(
            t2_out, size=target_size, mode="bilinear", align_corners=False
        )
        t3_resized = F.interpolate(
            t3_out, size=target_size, mode="bilinear", align_corners=False
        )

        # Concatenate features along channel dimension
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)

        # 4. Fusion convolution
        fused = F.relu(self.bn_fusion(self.fusion_conv(hypercolumn)))

        # 5. CBAM attention
        attended = self.cbam(fused)

        # 6. Global pooling and classification
        out = self.avgpool(attended)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Extract intermediate feature maps for visualization.
        
        Useful for Grad-CAM and other explainability methods.
        
        Returns:
            Dictionary with feature maps from each stage.
        """
        features = {}
        
        x = self.init_conv(x)
        features["init"] = x

        x = self.db1(x)
        t1_out = self.t1(x)
        features["t1"] = t1_out

        x = self.db2(t1_out)
        t2_out = self.t2(x)
        features["t2"] = t2_out

        x = self.db3(t2_out)
        t3_out = self.t3(x)
        features["t3"] = t3_out

        x = self.db4(t3_out)
        x_final = self.norm_final(x)
        features["final"] = x_final

        return features
