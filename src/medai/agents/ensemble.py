"""
Ensemble Agent for MedAI.

Multi-model ensemble for robust fracture classification using weighted voting.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from PIL import Image

from medai.config import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, DEVICE, MODELS_DIR
from medai.models.factory import load_model_from_checkpoint
from medai.utils.transforms import get_transforms

__all__ = ["EnsembleAgent"]


class EnsembleAgent:
    """
    Multi-model ensemble agent for fracture classification.
    
    Combines predictions from multiple models using weighted voting
    for more robust and accurate predictions.
    
    Args:
        model_names: List of model architecture names to load.
        checkpoints_dir: Directory containing model checkpoints.
        num_classes: Number of classification classes.
        class_names: List of class names.
        device: Compute device (auto-detected if None).
        
    Examples:
        >>> agent = EnsembleAgent(
        ...     model_names=['swin', 'densenet169', 'maxvit'],
        ...     checkpoints_dir='models/',
        ... )
        >>> result = agent.run_ensemble('xray_image.jpg')
        >>> print(result['ensemble_prediction'], result['ensemble_confidence'])
    """
    
    def __init__(
        self,
        model_names: List[str],
        checkpoints_dir: Optional[str] = None,
        num_classes: int = NUM_CLASSES,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or DEVICE
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.model_names = model_names
        self.transforms = get_transforms("val", IMG_SIZE)
        
        # Use default models directory if not specified
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else MODELS_DIR
        
        # Load all models
        self.models: Dict[str, torch.nn.Module] = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all specified model checkpoints."""
        print(f"Loading {len(self.model_names)} models from {self.checkpoints_dir}")
        
        for name in self.model_names:
            checkpoint_path = self.checkpoints_dir / f"best_{name}.pth"
            
            # Try alternative naming patterns
            if not checkpoint_path.exists():
                alt_patterns = [
                    f"best_{name}v2.pth",
                    f"{name}.pth",
                    f"best_hypercolumn_cbam_densenet169.pth" if "hypercolumn" in name else None,
                ]
                for pattern in alt_patterns:
                    if pattern:
                        alt_path = self.checkpoints_dir / pattern
                        if alt_path.exists():
                            checkpoint_path = alt_path
                            break
            
            if not checkpoint_path.exists():
                print(f"  ❌ Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                model = load_model_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model_name=name,
                    num_classes=self.num_classes,
                    device=self.device,
                    strict=False,  # Allow partial loading
                )
                self.models[name] = model
                print(f"  ✅ Loaded {name}")
            except Exception as e:
                print(f"  ❌ Failed to load {name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models were successfully loaded.")
        
        print(f"Successfully loaded {len(self.models)} models")
    
    @torch.no_grad()
    def run_ensemble(self, image_path: str) -> Dict[str, Any]:
        """
        Run ensemble inference on a single image.
        
        Uses weighted voting where weights are based on model confidence.
        
        Args:
            image_path: Path to the X-ray image.
            
        Returns:
            Dictionary containing:
                - image_path: Input image path
                - ensemble_prediction: Final predicted class
                - ensemble_confidence: Confidence of ensemble prediction
                - individual_predictions: Dict of predictions per model
                - fracture_detected: Boolean indicating fracture presence
                - all_probabilities: Weighted average probabilities
        """
        # Load and preprocess image
        image_path = Path(image_path).resolve()
        
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}
        
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {"error": f"Failed to process image: {e}"}
        
        # Collect predictions from all models
        all_probs = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            
            pred_idx = np.argmax(probs)
            individual_predictions[name] = {
                "class": self.class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": probs.tolist(),
            }
        
        # Weighted ensemble voting
        weights = np.array([np.max(probs) for probs in all_probs])
        weights = weights / np.sum(weights)
        
        weighted_avg_probs = np.average(all_probs, axis=0, weights=weights)
        ensemble_idx = np.argmax(weighted_avg_probs)
        ensemble_confidence = weighted_avg_probs[ensemble_idx]
        ensemble_class = self.class_names[ensemble_idx]
        
        return {
            "image_path": str(image_path),
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": float(ensemble_confidence),
            "ensemble_idx": int(ensemble_idx),
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": weighted_avg_probs.tolist(),
            "model_weights": weights.tolist(),
        }
    
    def run_ensemble_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Run ensemble on multiple images."""
        return [self.run_ensemble(path) for path in image_paths]
    
    def get_loaded_models(self) -> List[str]:
        """Return list of successfully loaded model names."""
        return list(self.models.keys())


def main():
    """CLI entry point for ensemble agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ensemble agent on an image")
    parser.add_argument("--image-path", required=True, help="Path to image")
    parser.add_argument("--checkpoints-dir", default=str(MODELS_DIR))
    parser.add_argument(
        "--models",
        default="swin,densenet169,efficientnetv2,maxvit,mobilenetv2",
        help="Comma-separated model names",
    )
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    
    args = parser.parse_args()
    
    model_names = [m.strip() for m in args.models.split(",")]
    
    agent = EnsembleAgent(
        model_names=model_names,
        checkpoints_dir=args.checkpoints_dir,
        num_classes=args.num_classes,
    )
    
    result = agent.run_ensemble(args.image_path)
    
    print("\n--- ENSEMBLE RESULTS ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['ensemble_prediction']}")
        print(f"Confidence: {result['ensemble_confidence']:.4f}")
        print(f"Fracture Detected: {'YES' if result['fracture_detected'] else 'NO'}")
        print("\nIndividual Predictions:")
        for name, pred in result["individual_predictions"].items():
            print(f"  {name}: {pred['class']} ({pred['confidence']:.4f})")
    print("-" * 30)


if __name__ == "__main__":
    main()
