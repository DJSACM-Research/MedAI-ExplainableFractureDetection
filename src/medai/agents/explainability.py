"""
Explainability Agent for MedAI.

Generates textual explanations for model predictions using Grad-CAM heatmaps.
"""

from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from medai.config import CLASS_NAMES

__all__ = [
    "ExplainabilityAgent",
    "calculate_heatmap_centroid",
    "generate_random_heatmap",
]


def calculate_heatmap_centroid(
    cam_array: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate the centroid (center of mass) of significant activation in a heatmap.
    
    Args:
        cam_array: 2D array of Grad-CAM activations (normalized 0-1).
        threshold: Activation threshold for identifying significant regions.
        
    Returns:
        Tuple of (normalized_x, normalized_y, max_activation).
        Coordinates are normalized to [0, 1].
    """
    # Apply threshold
    binary_map = cam_array > threshold
    
    if not np.any(binary_map):
        return (0.5, 0.5, 0.0)
    
    # Get coordinates and weights
    coords = np.argwhere(binary_map)
    weights = cam_array[binary_map]
    
    if len(weights) == 0:
        return (0.5, 0.5, 0.0)
    
    # Calculate weighted centroid
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    sum_weights = np.sum(weights)
    
    centroid_x = np.sum(x_coords * weights) / sum_weights
    centroid_y = np.sum(y_coords * weights) / sum_weights
    
    # Normalize to [0, 1]
    h, w = cam_array.shape
    norm_x = centroid_x / w
    norm_y = centroid_y / h
    max_activation = np.max(weights)
    
    return (norm_x, norm_y, max_activation)


def generate_random_heatmap(size: int = 224) -> np.ndarray:
    """
    Generate a random plausible heatmap for testing.
    
    Args:
        size: Size of the square heatmap.
        
    Returns:
        2D numpy array with activation values (0-1).
    """
    cam_array = np.zeros((size, size), dtype=np.float32)
    
    # Random activation zone
    center_y = np.random.randint(size // 4, size * 3 // 4)
    center_x = np.random.randint(size // 4, size * 3 // 4)
    height = np.random.randint(30, 80)
    width = np.random.randint(30, 80)
    
    # Define bounds
    y_min = max(0, center_y - height // 2)
    y_max = min(size, center_y + height // 2)
    x_min = max(0, center_x - width // 2)
    x_max = min(size, center_x + width // 2)
    
    # Apply activation
    strength = np.random.uniform(0.6, 1.0)
    cam_array[y_min:y_max, x_min:x_max] = strength
    
    # Add noise
    cam_array = cam_array + np.random.uniform(0, 0.1, (size, size))
    cam_array = np.clip(cam_array, 0, 1)
    
    return cam_array


class ExplainabilityAgent:
    """
    Agent for generating textual explanations of model predictions.
    
    Converts Grad-CAM heatmaps and prediction results into human-readable
    explanations of what the model detected and where.
    
    Args:
        class_names: List of class names.
        body_part: Description of the body part being analyzed.
        
    Examples:
        >>> agent = ExplainabilityAgent(class_names=CLASS_NAMES)
        >>> explanation = agent.generate_explanation(diagnosis_result, cam_array)
        >>> print(explanation)
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        body_part: str = "bone",
    ):
        self.class_names = class_names or CLASS_NAMES
        self.body_part = body_part
    
    def generate_explanation(
        self,
        diagnosis_result: Dict[str, Any],
        cam_array: np.ndarray,
        threshold: float = 0.4,
    ) -> str:
        """
        Generate a textual explanation from diagnosis result and heatmap.
        
        Args:
            diagnosis_result: Output from DiagnosticAgent.
            cam_array: Grad-CAM heatmap array.
            threshold: Threshold for centroid calculation.
            
        Returns:
            Human-readable explanation string.
        """
        predicted_class = diagnosis_result.get("predicted_class", "Unknown")
        confidence = diagnosis_result.get("confidence_score", 0.0)
        
        # Analyze heatmap
        norm_x, norm_y, strength = calculate_heatmap_centroid(cam_array, threshold)
        
        # Determine location descriptions
        x_loc = "right side" if norm_x > 0.65 else ("left side" if norm_x < 0.35 else "center")
        y_loc = "distal end" if norm_y > 0.65 else ("proximal end" if norm_y < 0.35 else "middle region")
        
        # Generate explanation based on prediction
        if predicted_class == "Healthy":
            if confidence > 0.90:
                return (
                    f"The {self.body_part} appears **healthy** with high confidence "
                    f"({confidence:.2f}). No fracture pattern was detected."
                )
            else:
                return (
                    f"The {self.body_part} is likely **healthy** ({confidence:.2f}), "
                    f"though there is some low activation in the {y_loc} of the {x_loc} "
                    "that warrants a closer look."
                )
        
        if not diagnosis_result.get("fracture_detected", True):
            return "Diagnosis is **inconclusive** or data is missing."
        
        # Build fracture explanation
        intro = f"A fracture pattern consistent with a **{predicted_class}** type is detected"
        
        # Strength description
        if strength > 0.7:
            strength_adj = "strong"
        elif strength > 0.5:
            strength_adj = "clear"
        else:
            strength_adj = "mild"
        
        confidence_stmt = f"(Confidence: {confidence:.2f})"
        location_stmt = f"near the **{y_loc}** of the {self.body_part} in the {x_loc}."
        
        explanation = f"{intro} {confidence_stmt}. The model's focus is {strength_adj} {location_stmt}"
        
        # Add type-specific notes
        if predicted_class in ["Transverse", "Oblique"]:
            explanation += " This is based on a distinct linear focus."
        elif predicted_class == "Spiral":
            explanation += " The activation pattern suggests a twisting injury mechanism."
        elif predicted_class == "Comminuted":
            explanation += " Multiple activation foci indicate fragmented bone structure."
        
        return explanation
    
    def generate_summary(
        self,
        diagnosis_result: Dict[str, Any],
        cam_array: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate a structured summary with explanation components.
        
        Returns:
            Dictionary with explanation, location info, and confidence details.
        """
        explanation = self.generate_explanation(diagnosis_result, cam_array)
        norm_x, norm_y, strength = calculate_heatmap_centroid(cam_array)
        
        return {
            "explanation": explanation,
            "predicted_class": diagnosis_result.get("predicted_class"),
            "confidence": diagnosis_result.get("confidence_score"),
            "heatmap_centroid": {"x": norm_x, "y": norm_y},
            "activation_strength": strength,
            "fracture_detected": diagnosis_result.get("fracture_detected", False),
        }
