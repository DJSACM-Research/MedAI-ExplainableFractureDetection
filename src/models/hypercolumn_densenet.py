import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) for CBAM."""
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) for CBAM."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average and max pooling across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate the pooled outputs
        x_concat = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid
        out = self.conv(x_concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """Full CBAM module: CAM followed by SAM."""
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. Channel Attention
        x_ca = x * self.ca(x)
        # 2. Spatial Attention
        x_sa = x_ca * self.sa(x_ca)
        return x_sa

class HyperColumnCBAMDenseNet169(nn.Module):
    """
    Combines DenseNet169 (Backbone) with HyperColumns (Multi-scale Context) and CBAM (Attention).
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(HyperColumnCBAMDenseNet169, self).__init__()

        # Load pre-trained DenseNet169 backbone
        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet169(weights=weights)

        # Separate the feature extractor (Dense Blocks) from the classifier (final linear layer)
        self.features = densenet.features

        # --- Define Sequential Blocks for Chaining in Forward Pass ---
        self.init_conv = nn.Sequential(self.features.conv0, self.features.norm0, self.features.relu0, self.features.pool0)
        self.db1 = self.features.denseblock1
        self.t1 = self.features.transition1  # HC Source 1 (Output channels: 128)
        self.db2 = self.features.denseblock2
        self.t2 = self.features.transition2  # HC Source 2 (Output channels: 256)
        self.db3 = self.features.denseblock3
        self.t3 = self.features.transition3  # HC Source 3 (Output channels: 640)
        self.db4 = self.features.denseblock4
        self.norm_final = self.features.norm5 # Final normalization layer

        # Calculate the total number of channels after concatenating the HyperColumns
        HC_FUSION_CHANNELS = (self.norm_final.num_features +  # 1664
                              self.t3.conv.out_channels +      # 640
                              self.t2.conv.out_channels +      # 256
                              self.t1.conv.out_channels)       # 128
        # Total: 2688

        # Fusion layer to process the combined HyperColumn features
        self.fusion_conv = nn.Conv2d(HC_FUSION_CHANNELS, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)

        # Apply CBAM to the fused features
        self.cbam = CBAM(1024)

        # Global pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes) # Map fused channels to the number of classes
        )

    def forward(self, x):

        # 1. Initial layers
        x = self.init_conv(x)

        # 2. Dense Block 1 -> Transition 1 (HC Source 1)
        x = self.db1(x)
        t1_out = self.t1(x)

        # 3. Dense Block 2 -> Transition 2 (HC Source 2)
        x = self.db2(t1_out)
        t2_out = self.t2(x)

        # 4. Dense Block 3 -> Transition 3 (HC Source 3)
        x = self.db3(t2_out)
        t3_out = self.t3(x)

        # 5. Final Dense Block 4 -> Final Norm (HC Source 4)
        x = self.db4(t3_out)
        x_final = self.norm_final(x)

        # --- HyperColumn FUSION ---
        # Upsample all intermediate features to the size of the final feature map (x_final)
        upsample_target_size = x_final.shape[2:]

        # We use the outputs of the Transition layers for the HC features
        t1_resized = F.interpolate(t1_out, size=upsample_target_size, mode='bilinear', align_corners=False)
        t2_resized = F.interpolate(t2_out, size=upsample_target_size, mode='bilinear', align_corners=False)
        t3_resized = F.interpolate(t3_out, size=upsample_target_size, mode='bilinear', align_corners=False)

        # 2. Concatenate the features along the channel dimension
        hypercolumn_features = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)

        # 3. Fusion Convolution to reduce channel count
        fused_features = F.relu(self.bn_fusion(self.fusion_conv(hypercolumn_features)))

        # 4. Apply CBAM Attention
        attention_output = self.cbam(fused_features)

        # 5. Global Pooling
        out = self.avgpool(attention_output)

        # 6. Final Classification
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
