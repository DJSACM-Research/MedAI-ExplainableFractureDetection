"""
Image transformation utilities for MedAI.

Provides standardized transforms for training, validation, and inference.
"""

from typing import Tuple, List

import torchvision.transforms as T

__all__ = [
    "get_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_inverse_normalize",
]

# ImageNet normalization constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_transforms(
    split: str,
    img_size: int = 224,
    augmentation_strength: str = "standard"
) -> T.Compose:
    """
    Returns appropriate image transforms for the given split.
    
    Args:
        split: One of 'train', 'val', 'test', or 'inference'.
        img_size: Target image size (default: 224).
        augmentation_strength: Strength of augmentation for training.
            Options: 'light', 'standard', 'strong'.
    
    Returns:
        torchvision.transforms.Compose: Composed transforms.
        
    Examples:
        >>> train_transform = get_transforms('train', img_size=224)
        >>> val_transform = get_transforms('val', img_size=224)
    """
    if split == "train":
        return _get_train_transforms(img_size, augmentation_strength)
    else:
        return _get_eval_transforms(img_size)


def _get_train_transforms(img_size: int, strength: str) -> T.Compose:
    """Build training transforms with configurable augmentation strength."""
    
    # Base resize with slight padding for random crop
    resize_size = int(img_size * 1.1)
    
    if strength == "light":
        return T.Compose([
            T.Resize((resize_size, resize_size)),
            T.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    elif strength == "strong":
        return T.Compose([
            T.Resize((resize_size, resize_size)),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomRotation(20),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    else:  # standard
        return T.Compose([
            T.Resize((resize_size, resize_size)),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _get_eval_transforms(img_size: int) -> T.Compose:
    """Build evaluation/inference transforms (deterministic)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inverse_normalize() -> T.Compose:
    """
    Returns inverse normalization transform for visualization.
    
    Useful for converting normalized tensors back to displayable images.
    
    Returns:
        torchvision.transforms.Compose: Inverse normalization transform.
        
    Examples:
        >>> inverse_norm = get_inverse_normalize()
        >>> displayable_tensor = inverse_norm(normalized_tensor)
    """
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1 / s for s in IMAGENET_STD]
    
    return T.Compose([
        T.Normalize(mean=[0, 0, 0], std=inv_std),
        T.Normalize(mean=inv_mean, std=[1, 1, 1]),
    ])


def get_tta_transforms(img_size: int = 224) -> List[T.Compose]:
    """
    Returns a list of transforms for Test-Time Augmentation (TTA).
    
    Args:
        img_size: Target image size.
        
    Returns:
        List of transform compositions for TTA.
    """
    base_normalize = [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    
    return [
        # Original
        T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            *base_normalize,
        ]),
        # Horizontal flip
        T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=1.0),
            *base_normalize,
        ]),
        # Slight rotation
        T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.RandomRotation(degrees=(5, 5)),
            *base_normalize,
        ]),
        # Slight zoom
        T.Compose([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            T.CenterCrop(img_size),
            *base_normalize,
        ]),
    ]
