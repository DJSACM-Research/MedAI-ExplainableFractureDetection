"""
Test configuration and fixtures for the MedAI Explainable Fracture Detection system.

This module provides common test utilities, fixtures, and configuration
for all test suites.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# --- Test Constants ---

CLASS_NAMES = [
    "Comminuted", 
    "Greenstick", 
    "Healthy", 
    "Oblique",
    "Oblique Displaced", 
    "Spiral", 
    "Transverse", 
    "Transverse Displaced"
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
DEVICE = "cpu"

# Standard knowledge base for testing
KNOWLEDGE_BASE = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more pieces.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": ["Usually requires surgical intervention (ORIF).", 
                               "Long immobilization time (8-12 weeks).", 
                               "Requires physical therapy."],
        "prognosis_notes": "Risk of non-union is higher. Full recovery may take 6+ months."
    },
    "Greenstick": {
        "definition": "A fracture where the bone is cracked but not completely broken through.",
        "icd_code": "S52.1",
        "severity": "Low-Medium",
        "treatment_guidelines": ["Immobilization for 4-6 weeks."],
        "prognosis_notes": "Excellent prognosis with proper care."
    },
    "Healthy": {
        "definition": "No evidence of fracture.",
        "icd_code": "Z00.0",
        "severity": "Low",
        "treatment_guidelines": ["No treatment required."],
        "prognosis_notes": "Normal bone health."
    },
    "Oblique": {
        "definition": "A clean break at an angle.",
        "icd_code": "S52.2",
        "severity": "Moderate",
        "treatment_guidelines": ["Reduction and immobilization for 6-8 weeks."],
        "prognosis_notes": "Good prognosis with proper alignment."
    },
    "Oblique Displaced": {
        "definition": "A bone broken at an angle with pieces shifted out of place.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": ["Requires reduction (closed or open).", 
                               "Often requires casting or sometimes surgery."],
        "prognosis_notes": "Good prognosis if successfully reduced and stabilized."
    },
    "Spiral": {
        "definition": "A twisting break that spirals around the bone.",
        "icd_code": "S52.3",
        "severity": "Serious",
        "treatment_guidelines": ["Usually requires surgery.", 
                               "6-12 weeks immobilization."],
        "prognosis_notes": "Good prognosis with proper treatment."
    },
    "Transverse": {
        "definition": "A clean break straight across the bone.",
        "icd_code": "S52.2",
        "severity": "Moderate",
        "treatment_guidelines": ["Reduction and immobilization for 6-8 weeks."],
        "prognosis_notes": "Good prognosis with proper alignment."
    },
    "Transverse Displaced": {
        "definition": "A bone broken straight across with pieces shifted out of place.",
        "icd_code": "S52.9",
        "severity": "Serious",
        "treatment_guidelines": ["Requires reduction and stabilization.", 
                               "Often requires surgery."],
        "prognosis_notes": "Good prognosis with proper reduction and immobilization."
    }
}


# --- Test Image Generation ---

def create_test_image(width: int = 224, height: int = 224, color: str = 'red') -> str:
    """
    Create a temporary test image file.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: PIL color name for the image
        
    Returns:
        Path to the temporary image file
    """
    img = Image.new('RGB', (width, height), color=color)
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name)
    return temp_file.name


def create_test_heatmap(size: int = 224, activation_strength: float = 0.8) -> np.ndarray:
    """
    Create a test heatmap array.
    
    Args:
        size: Size of the heatmap (will be size x size)
        activation_strength: Strength of the activation (0-1)
        
    Returns:
        Numpy array representing a heatmap
    """
    heatmap = np.zeros((size, size), dtype=np.float32)
    # Add a centered activation zone
    center = size // 2
    zone_size = size // 4
    heatmap[center-zone_size:center+zone_size, center-zone_size:center+zone_size] = activation_strength
    return heatmap


# --- Mock Diagnosis Results ---

def get_mock_healthy_diagnosis() -> dict:
    """Get a mock healthy bone diagnosis result."""
    return {
        "image_path": "test_healthy.jpg",
        "fracture_detected": False,
        "predicted_class": "Healthy",
        "severity_type": "Healthy",
        "confidence_score": 0.95,
        "uncertainty_score": 0.05,
        "all_probabilities": [0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
    }


def get_mock_fracture_diagnosis(fracture_type: str = "Spiral") -> dict:
    """
    Get a mock fracture diagnosis result.
    
    Args:
        fracture_type: Type of fracture to simulate
        
    Returns:
        Mock diagnosis dictionary
    """
    class_idx = CLASS_NAMES.index(fracture_type)
    probs = [0.05] * NUM_CLASSES
    probs[class_idx] = 0.75
    
    return {
        "image_path": f"test_{fracture_type.lower()}.jpg",
        "fracture_detected": True,
        "predicted_class": fracture_type,
        "severity_type": fracture_type,
        "confidence_score": 0.75,
        "uncertainty_score": 0.25,
        "all_probabilities": probs
    }


def get_mock_ensemble_result(prediction: str = "Healthy") -> dict:
    """
    Get a mock ensemble agent result.
    
    Args:
        prediction: The ensemble prediction class name
        
    Returns:
        Mock ensemble result dictionary
    """
    return {
        "image_path": "test_ensemble.jpg",
        "ensemble_prediction": prediction,
        "ensemble_confidence": 0.87,
        "individual_predictions": {
            "swin": {"class": prediction, "confidence": 0.85},
            "mobilenetv2": {"class": prediction, "confidence": 0.90},
            "densenet169": {"class": prediction, "confidence": 0.87}
        },
        "fracture_detected": prediction != "Healthy"
    }


# --- Test Utilities ---

def cleanup_temp_file(filepath: str) -> None:
    """
    Clean up a temporary file.
    
    Args:
        filepath: Path to the file to delete
    """
    if os.path.exists(filepath):
        os.unlink(filepath)
