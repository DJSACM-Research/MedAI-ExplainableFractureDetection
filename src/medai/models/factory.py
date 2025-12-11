"""
Model Factory for MedAI.

Provides functions for creating and loading models with various architectures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import timm
import torchvision.models as tvmodels

from medai.models.architectures import HyperColumnCBAMDenseNet169

__all__ = [
    "get_model",
    "load_model_from_checkpoint",
    "get_available_models",
]

# Registry of supported models
MODEL_REGISTRY: Dict[str, Dict] = {
    "hypercolumn_densenet169": {
        "class": "custom",
        "description": "HyperColumn DenseNet169 with CBAM attention",
    },
    "swin": {
        "timm_name": "swin_small_patch4_window7_224",
        "description": "Swin Transformer Small",
    },
    "convnext": {
        "timm_name": "convnext_tiny",
        "description": "ConvNeXt Tiny",
    },
    "densenet169": {
        "class": "torchvision",
        "description": "DenseNet-169",
    },
    "mobilenetv2": {
        "timm_name": "mobilenetv2_100",
        "description": "MobileNetV2",
    },
    "efficientnetv2": {
        "timm_name": "efficientnet_b0",
        "description": "EfficientNet-B0",
    },
    "maxvit": {
        "timm_name": "maxvit_tiny_tf_224",
        "description": "MaxViT Tiny",
    },
}


def get_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Create a model architecture by name.
    
    Args:
        name: Model architecture name. Options:
            - 'hypercolumn_densenet169': Custom HyperColumn with CBAM
            - 'swin': Swin Transformer Small
            - 'convnext': ConvNeXt Tiny
            - 'densenet169': DenseNet-169
            - 'mobilenetv2': MobileNetV2
            - 'efficientnetv2': EfficientNet-B0
            - 'maxvit': MaxViT Tiny
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        dropout: Dropout probability (for applicable models).
        
    Returns:
        nn.Module: The model instance.
        
    Raises:
        ValueError: If model name is not recognized.
        
    Examples:
        >>> model = get_model('swin', num_classes=8)
        >>> model = get_model('hypercolumn_densenet169', num_classes=8)
    """
    name = name.lower().strip()
    
    # Handle aliases
    name_aliases = {
        "hypercolumn": "hypercolumn_densenet169",
        "hypercolumn_cbam": "hypercolumn_densenet169",
        "densenet": "densenet169",
        "mobilenet": "mobilenetv2",
        "efficientnet": "efficientnetv2",
    }
    name = name_aliases.get(name, name)
    
    if name == "hypercolumn_densenet169":
        return HyperColumnCBAMDenseNet169(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    
    if name == "densenet169":
        model = tvmodels.densenet169(
            weights=tvmodels.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    
    # Try to load from timm
    if name in MODEL_REGISTRY and "timm_name" in MODEL_REGISTRY[name]:
        timm_name = MODEL_REGISTRY[name]["timm_name"]
        model = timm.create_model(timm_name, pretrained=pretrained)
        
        # Adapt classifier head
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes=num_classes)
        elif hasattr(model, "head"):
            if hasattr(model.head, "fc"):
                model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
            else:
                model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, "classifier"):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise RuntimeError(f"Cannot adapt classifier for model: {name}")
            
        return model
    
    raise ValueError(
        f"Unknown model: '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
    )


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str,
    num_classes: int,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model_name: Architecture name for creating the model.
        num_classes: Number of output classes.
        device: Device to load the model to.
        strict: Whether to strictly enforce state dict matching.
        
    Returns:
        nn.Module: The loaded model in evaluation mode.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If state dict loading fails.
        
    Examples:
        >>> model = load_model_from_checkpoint(
        ...     'models/best_swin.pth',
        ...     model_name='swin',
        ...     num_classes=8,
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        from medai.utils.device import get_device
        device = get_device()
    
    # Create model architecture
    model = get_model(model_name, num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Clean up state dict keys if needed (remove 'module.' prefix from DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    
    return model


def get_available_models() -> List[str]:
    """
    Get list of available model architectures.
    
    Returns:
        List of model names.
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> Dict:
    """
    Get information about a model architecture.
    
    Args:
        name: Model name.
        
    Returns:
        Dictionary with model information.
    """
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]
