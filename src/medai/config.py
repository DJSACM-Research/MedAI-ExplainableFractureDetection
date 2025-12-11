"""
MedAI Configuration Module

Centralized configuration for all constants, paths, and settings.
Environment variables override default values.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import torch

# ============================================================================
# Path Configuration
# ============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "balanced_augmented_dataset"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Temporary directories
TEMP_DIR = PROJECT_ROOT / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# ============================================================================
# Device Configuration
# ============================================================================

def get_device() -> torch.device:
    """
    Dynamically selects the best available device: CUDA, MPS, or CPU.
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def require_mps() -> torch.device:
    """
    Enforces MPS device (for Mac-only scripts).
    
    Returns:
        torch.device: MPS device.
        
    Raises:
        RuntimeError: If MPS is not available.
    """
    if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Apple Silicon) is required but not available.")
    return torch.device("mps")


DEVICE = get_device()

# ============================================================================
# Class Configuration
# ============================================================================

CLASS_NAMES: List[str] = [
    "Comminuted",
    "Greenstick",
    "Healthy",
    "Oblique",
    "Oblique Displaced",
    "Spiral",
    "Transverse",
    "Transverse Displaced",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# Class name to index mapping
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ============================================================================
# Image Configuration
# ============================================================================

IMG_SIZE: int = int(os.getenv("MEDAI_IMG_SIZE", "224"))
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ============================================================================
# Model Configuration
# ============================================================================

# Available model architectures and their checkpoint filenames
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hypercolumn_cbam": {
        "checkpoint": "best_hypercolumn_cbam_densenet169.pth",
        "architecture": "hypercolumn_densenet169",
        "description": "HyperColumn with CBAM attention on DenseNet169 backbone",
    },
    "swin": {
        "checkpoint": "best_swin.pth",
        "architecture": "swin",
        "description": "Swin Transformer Small",
    },
    "densenet169": {
        "checkpoint": "best_densenet169.pth",
        "architecture": "densenet169",
        "description": "DenseNet-169",
    },
    "efficientnet": {
        "checkpoint": "best_efficientnetv2.pth",
        "architecture": "efficientnetv2",
        "description": "EfficientNet-B0",
    },
    "mobilenetv2": {
        "checkpoint": "best_mobilenetv2.pth",
        "architecture": "mobilenetv2",
        "description": "MobileNetV2",
    },
    "maxvit": {
        "checkpoint": "best_maxvit.pth",
        "architecture": "maxvit",
        "description": "MaxViT Tiny",
    },
}

# Default ensemble models (top performers)
DEFAULT_ENSEMBLE_MODELS: List[str] = [
    "maxvit",
    "densenet169",
    "hypercolumn_cbam",
    "efficientnet",
    "mobilenetv2",
    "swin",
]

# ============================================================================
# Training Configuration
# ============================================================================

TRAINING_DEFAULTS: Dict[str, Any] = {
    "batch_size": 32,
    "num_workers": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 20,
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.4,
    "early_stopping_patience": 5,
}

# ============================================================================
# LLM Configuration (for Patient Interaction)
# ============================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

HUGGINGFACE_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models"

# ============================================================================
# Medical Knowledge Base
# ============================================================================

MEDICAL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more pieces.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": [
            "Usually requires surgical intervention (ORIF - Open Reduction Internal Fixation).",
            "Long immobilization time (8-12 weeks).",
            "Requires physical therapy.",
        ],
        "prognosis_notes": "Risk of non-union is higher. Full recovery may take 6+ months.",
    },
    "Greenstick": {
        "definition": "A partial fracture where the bone is cracked but not completely broken through. Common in children.",
        "icd_code": "S52.3",
        "severity": "Low-Moderate",
        "treatment_guidelines": [
            "Immobilization with cast or splint.",
            "Careful monitoring for progression.",
            "Minimal surgical intervention usually needed.",
        ],
        "prognosis_notes": "Generally good prognosis. Recovery typically within 4-6 weeks.",
    },
    "Healthy": {
        "definition": "No evidence of fracture. Bone appears normal.",
        "icd_code": "Z00.0",
        "severity": "None",
        "treatment_guidelines": [
            "No treatment required.",
            "Continue normal activities as tolerated.",
            "Regular follow-up if there is persistent pain.",
        ],
        "prognosis_notes": "Normal bone health. No intervention needed.",
    },
    "Oblique": {
        "definition": "A diagonal break across the bone at approximately 45 degrees.",
        "icd_code": "S52.2",
        "severity": "Moderate",
        "treatment_guidelines": [
            "Immobilization with cast or splint.",
            "Regular X-rays to monitor healing.",
            "Physical therapy after immobilization period.",
        ],
        "prognosis_notes": "Good prognosis with proper immobilization. Recovery typically 6-8 weeks.",
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are not aligned and have shifted out of place.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": [
            "Requires reduction (closed or open).",
            "Often requires casting or sometimes surgery to stabilize.",
            "Regular X-rays to ensure proper alignment.",
        ],
        "prognosis_notes": "Good prognosis if successfully reduced and stabilized. Recovery 8-12 weeks.",
    },
    "Spiral": {
        "definition": "A twisting break that spirals around the bone, typically caused by rotational forces.",
        "icd_code": "S52.4",
        "severity": "Serious",
        "treatment_guidelines": [
            "Usually requires immobilization in a cast or brace.",
            "May require surgery if fragments are unstable.",
            "Requires extensive physical therapy.",
        ],
        "prognosis_notes": "Variable recovery time. May take 8-16 weeks depending on severity.",
    },
    "Transverse": {
        "definition": "A clean break straight across the bone, perpendicular to the bone's long axis.",
        "icd_code": "S52.1",
        "severity": "Moderate",
        "treatment_guidelines": [
            "Immobilization with cast or splint.",
            "Regular X-rays to monitor alignment.",
            "Physical therapy after healing begins.",
        ],
        "prognosis_notes": "Good prognosis. Clean breaks typically heal well. Recovery 6-10 weeks.",
    },
    "Transverse Displaced": {
        "definition": "A straight break across the bone with fragments shifted out of place.",
        "icd_code": "S52.8",
        "severity": "Serious",
        "treatment_guidelines": [
            "Requires reduction (closed or open).",
            "Often requires surgery to realign fragments.",
            "Long-term immobilization and rehabilitation.",
        ],
        "prognosis_notes": "Good prognosis with treatment. Recovery 10-14 weeks.",
    },
}

# ============================================================================
# Streamlit Configuration
# ============================================================================

STREAMLIT_CONFIG: Dict[str, Any] = {
    "page_title": "🦴 MedAI Fracture Detection",
    "page_icon": "🦴",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
