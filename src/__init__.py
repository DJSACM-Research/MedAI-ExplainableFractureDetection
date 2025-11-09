"""
Source code for MedAI Fracture Detection System.
"""

__version__ = "1.0.0"

# Import key utilities for easy access
from .utils import (
    get_device,
    require_mps,
    DEVICE,
    get_model,
    get_transforms,
    FractureDataset
)

__all__ = [
    'get_device',
    'require_mps',
    'DEVICE',
    'get_model',
    'get_transforms',
    'FractureDataset'
]
