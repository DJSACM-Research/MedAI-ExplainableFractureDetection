"""
MedAI: Explainable Fracture Detection System

A comprehensive medical imaging AI system for fracture detection with
multi-model ensemble, explainability, and patient education features.
"""

__version__ = "1.0.0"
__author__ = "Hardik"

from medai.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    IMG_SIZE,
    DEVICE,
    MODELS_DIR,
    DATA_DIR,
)

__all__ = [
    "__version__",
    "__author__",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "IMG_SIZE",
    "DEVICE",
    "MODELS_DIR",
    "DATA_DIR",
]
