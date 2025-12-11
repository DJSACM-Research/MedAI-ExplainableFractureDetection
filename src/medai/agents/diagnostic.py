"""
Diagnostic Agent for MedAI.

Performs single-model fracture classification on X-ray images.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image

from medai.config import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, DEVICE
from medai.models.factory import load_model_from_checkpoint
from medai.utils.transforms import get_transforms

__all__ = ["DiagnosticAgent"]


class DiagnosticAgent:
    """
    Agent for fracture diagnosis using a single trained model.
    
    Performs classification on X-ray images and returns detailed
    prediction results including confidence scores and uncertainty.
    
    Args:
        checkpoint_path: Path to the model checkpoint.
        model_name: Architecture name (e.g., 'swin', 'densenet169').
        num_classes: Number of classification classes.
        img_size: Input image size.
        class_names: List of class names.
        device: Compute device (auto-detected if None).
        
    Examples:
        >>> agent = DiagnosticAgent(
        ...     checkpoint_path='models/best_swin.pth',
        ...     model_name='swin',
        ... )
        >>> result = agent.run_diagnosis('xray_image.jpg')
        >>> print(result['predicted_class'], result['confidence_score'])
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        num_classes: int = NUM_CLASSES,
        img_size: int = IMG_SIZE,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or DEVICE
        self.img_size = img_size
        self.class_names = class_names or CLASS_NAMES
        self.model_name = model_name
        
        # Load model
        self.model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            num_classes=num_classes,
            device=self.device,
        )
        
        # Setup transforms
        self.transform = get_transforms("val", self.img_size)
        
        print(f"✅ DiagnosticAgent loaded {model_name} on {self.device}")
    
    def run_diagnosis(self, image_path: str) -> Dict[str, Any]:
        """
        Run fracture classification on a single image.
        
        Args:
            image_path: Path to the X-ray image.
            
        Returns:
            Dictionary containing:
                - image_path: Input image path
                - fracture_detected: Boolean indicating fracture presence
                - predicted_class: Predicted class name
                - predicted_idx: Predicted class index
                - confidence_score: Confidence of prediction (0-1)
                - uncertainty_score: 1 - confidence_score
                - all_probabilities: List of probabilities for each class
        """
        # Resolve path
        full_path = Path(image_path).resolve()
        
        if not full_path.exists():
            return {"error": f"Image file not found: {image_path}"}
        
        # Load and preprocess image
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to open image: {e}"}
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        
        # Extract predictions
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        predicted_class = self.class_names[predicted_idx]
        
        # Determine fracture presence
        is_fracture = predicted_class != "Healthy"
        
        return {
            "image_path": str(image_path),
            "fracture_detected": is_fracture,
            "predicted_class": predicted_class,
            "predicted_idx": predicted_idx,
            "severity_type": predicted_class,
            "confidence_score": confidence,
            "uncertainty_score": 1.0 - confidence,
            "all_probabilities": probabilities.cpu().numpy().tolist(),
        }
    
    def diagnose_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run diagnosis on multiple images.
        
        Args:
            image_paths: List of image paths.
            
        Returns:
            List of diagnosis results.
        """
        return [self.run_diagnosis(path) for path in image_paths]


def main():
    """CLI entry point for diagnostic agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run diagnostic agent on an image")
    parser.add_argument("--image-path", required=True, help="Path to image")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--model", default="swin", help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    
    args = parser.parse_args()
    
    agent = DiagnosticAgent(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size,
    )
    
    result = agent.run_diagnosis(args.image_path)
    
    print("\n--- DIAGNOSTIC RESULTS ---")
    if "error" in result:
        print(f"Status: FAILED\nReason: {result['error']}")
    else:
        print(f"Status: SUCCESS")
        print(f"Image: {result['image_path']}")
        print(f"Fracture Detected: {'YES' if result['fracture_detected'] else 'NO'}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence_score']:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
