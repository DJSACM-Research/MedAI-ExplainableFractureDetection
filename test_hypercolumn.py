"""Test script to verify HypercolumnCBAMDenseNet model loading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    """Single dense layer as used in DenseNet."""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class _DenseBlock(nn.ModuleDict):
    """Dense block containing multiple dense layers."""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Module):
    """Transition layer between dense blocks (no pooling for hypercolumn)."""
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        # Note: No pool layer - pooling handled separately or via stride
    
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM with shared MLP."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class HypercolumnCBAMDenseNet(nn.Module):
    """
    Custom DenseNet169 with Hypercolumn fusion and CBAM attention.
    Matches the exact architecture from training checkpoint.
    """
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(HypercolumnCBAMDenseNet, self).__init__()
        
        # DenseNet169 block config: [6, 12, 32, 32]
        block_config = (6, 12, 32, 32)
        num_init_features = 64
        
        # Initial convolution (features.conv0, features.norm0)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Add dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Custom hypercolumn components
        # init_conv: 7x7 Conv from RGB (3 channels) to 64 channels + BN
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Custom dense blocks for hypercolumn feature processing
        # db1: 6 layers, db2: 12 layers, db3: 32 layers, db4: 32 layers
        # (matching DenseNet169 config)
        self.db1 = _DenseBlock(num_layers=6, num_input_features=64, bn_size=bn_size, growth_rate=growth_rate)
        self.db2 = _DenseBlock(num_layers=12, num_input_features=128, bn_size=bn_size, growth_rate=growth_rate)
        self.db3 = _DenseBlock(num_layers=32, num_input_features=256, bn_size=bn_size, growth_rate=growth_rate)
        self.db4 = _DenseBlock(num_layers=32, num_input_features=640, bn_size=bn_size, growth_rate=growth_rate)
        
        # Transition layers for hypercolumn (no pooling - just norm + conv)
        # t1: 256 -> 128, t2: 512 -> 256, t3: 1280 -> 640
        self.t1 = _Transition(num_input_features=256, num_output_features=128)
        self.t2 = _Transition(num_input_features=512, num_output_features=256)
        self.t3 = _Transition(num_input_features=1280, num_output_features=640)
        
        # Final normalization (1664 channels from db4: 640 + 32*32)
        self.norm_final = nn.BatchNorm2d(1664)
        
        # Hypercolumn fusion: channels = 128 + 256 + 640 + 1664 = 2688
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        
        # CBAM attention with 1024 channels
        self.cbam = CBAM(1024)
        
        # Classifier: AdaptiveAvgPool (index 0) + Linear (index 1)
        # Note: classifier.1 in checkpoint means Linear is at index 1
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(1024, num_classes)  # Will be at index 1 after AdaptiveAvgPool
        )
    
    def forward(self, x):
        # Use init_conv to process raw RGB input (3 -> 64 channels)
        x = self.init_conv(x)  # 7x7 conv with stride 2 + BN
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # Process through custom dense blocks
        x1 = self.db1(x)  # 64 + 6*32 = 256 channels
        x1_t = self.t1(x1)  # 256 -> 128 channels
        
        x2 = self.db2(x1_t)  # 128 + 12*32 = 512 channels
        x2_t = self.t2(x2)  # 512 -> 256 channels
        
        x3 = self.db3(x2_t)  # 256 + 32*32 = 1280 channels
        x3_t = self.t3(x3)  # 1280 -> 640 channels
        
        x4 = self.db4(x3_t)  # 640 + 32*32 = 1664 channels
        x4 = self.norm_final(x4)
        
        # Upsample all to match x1_t size for hypercolumn
        target_size = x1_t.shape[2:]
        
        f1 = x1_t  # 128 channels
        f2 = F.interpolate(x2_t, size=target_size, mode='bilinear', align_corners=False)  # 256 channels
        f3 = F.interpolate(x3_t, size=target_size, mode='bilinear', align_corners=False)  # 640 channels
        f4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)  # 1664 channels
        
        # Concatenate hypercolumn features: 128 + 256 + 640 + 1664 = 2688
        hypercolumn = torch.cat([f1, f2, f3, f4], dim=1)
        
        # Fusion
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = F.relu(x)
        
        # Apply CBAM attention
        x = self.cbam(x)
        
        # Classify: AdaptiveAvgPool then flatten then Linear
        x = self.classifier[0](x)  # AdaptiveAvgPool2d
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier[1](x)  # Linear
        
        return x


if __name__ == "__main__":
    print("Creating HypercolumnCBAMDenseNet model...")
    model = HypercolumnCBAMDenseNet(num_classes=8)
    
    print("Loading checkpoint...")
    ckpt_path = 'models/best_hypercolumn_cbam_densenet169.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    print(f"\nCheckpoint keys (first 10): {list(state_dict.keys())[:10]}")
    print(f"Model keys (first 10): {list(model.state_dict().keys())[:10]}")
    
    result = model.load_state_dict(state_dict, strict=False)
    
    print(f"\nMissing keys ({len(result.missing_keys)}): {result.missing_keys[:20] if len(result.missing_keys) > 20 else result.missing_keys}")
    print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:20] if len(result.unexpected_keys) > 20 else result.unexpected_keys}")
    
    if not result.missing_keys and not result.unexpected_keys:
        print("\n✅ Model loaded perfectly!")
    elif not result.missing_keys:
        print("\n✅ Model loaded successfully (some extra keys in checkpoint ignored)")
    else:
        print("\n❌ Some model keys are missing!")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("✅ Inference successful!")
