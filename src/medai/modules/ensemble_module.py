import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import timm 
from medai.uncertainty.conformal import predict_conformal_set

# ----------------------------------------------------------------------
# --- Helper Functions (Duplicated for standalone capability) ---
# ----------------------------------------------------------------------

DEVICE = None
IMG_SIZE = 224

def get_device():
    """Detects and returns the appropriate torch device."""
    global DEVICE
    if DEVICE is None:
        if torch.cuda.is_available(): 
            DEVICE = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): 
            DEVICE = torch.device('mps')
        else:
            DEVICE = torch.device('cpu')
    return DEVICE

def get_model(name: str, num_classes: int, pretrained: bool=True):
    """Loads and adapts one of the specified pretrained models from timm."""
    name = name.lower()
    
    # Simple mapping, can be expanded
    model_map = {
        'swin': 'swin_small_patch4_window7_224',
        'mobilenetv2': 'mobilenetv2_100',
        'efficientnetv2': 'tf_efficientnetv2_s',
        'maxvit': 'maxvit_rmlp_small_rw_224',
        'densenet169': 'densenet169',
    }
    # Check for hypercolumn variants or exact matches
    if name in model_map:
        timm_name = model_map[name]
    elif name.startswith('swin'): timm_name = 'swin_small_patch4_window7_224'
    elif 'densenet' in name: timm_name = 'densenet169'
    elif 'efficientnet' in name: timm_name = 'tf_efficientnetv2_s'
    else:
        # Fallback: try to us name directly
        timm_name = name

    try:
        m = timm.create_model(timm_name, pretrained=pretrained)
    except Exception:
        raise ValueError(f"Unknown or unavailable model: {name}")
    
    # Adjust classifier head
    if hasattr(m, 'head') and isinstance(m.head, nn.Linear):
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif hasattr(m, 'classifier') and isinstance(m.classifier, nn.Linear):
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        try:
            m.reset_classifier(num_classes=num_classes)
        except Exception:
            # Some models might need custom logic
            pass

    return m

def get_transforms(img_size: int = 224):
    """Standard image transformations for inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _swap_prediction_label(label: str) -> str:
    """
    Swaps predictions for specific classes as requested:
    Transverse <-> Transverse Displaced
    Oblique <-> Oblique Displaced
    """
    if label == "Transverse":
        return "Transverse Displaced"
    elif label == "Transverse Displaced":
        return "Transverse"
    elif label == "Oblique":
        return "Oblique Displaced"
    elif label == "Oblique Displaced":
        return "Oblique"
    return label

# ----------------------------------------------------------------------
# --- Ensemble Module Core ---
# ----------------------------------------------------------------------

class EnsembleModule:
    """Runs inference across multiple models and combines predictions."""
    
    # Classes where hypercolumn models should get more weight
    HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique Displaced", "Transverse", "Transverse Displaced"}
    # Weight for hypercolumn models when priority class is detected
    HYPERCOLUMN_WEIGHT = 1.0
    # Weight for other models
    DEFAULT_WEIGHT = 1.0
    
    def __init__(self, 
                 class_names: List[str], 
                 models: Optional[Dict[str, nn.Module]] = None, 
                 model_names: Optional[List[str]] = None,
                 checkpoints_dir: Optional[str] = None,
                 num_classes: int = 8, 
                 device=None,
                 img_size: int = 224, 
                 conformal_threshold: float = None):
        
        self.class_names = class_names
        self.device = device if device else get_device()
        self.transforms = get_transforms(img_size)
        self.conformal_threshold = conformal_threshold
        self.num_classes = num_classes
        
        self.models = {}
        if models is not None:
             self.models = models
             # validation?
        elif model_names and checkpoints_dir:
            self.model_names = model_names
            self._load_all_models(checkpoints_dir)
        else:
            raise ValueError("Either 'models' dict OR ('model_names' list AND 'checkpoints_dir') must be provided.")
            
    def _load_all_models(self, checkpoints_dir: str):
        """Loads all specified model checkpoints."""
        print(f"Loading {len(self.model_names)} models from {checkpoints_dir} on {self.device}...")
        
        for name in self.model_names:
            checkpoint_path = os.path.join(checkpoints_dir, f"best_{name}.pth")
            try:
                # Need robust model loading here. 
                # Assuming generic names like 'densenet169' work with get_model logic
                # For hypercolumn models, the name might be complex e.g. 'hypercolumn_densenet169'
                # get_model needs to handle it or map it.
                # For simplicity here, we assume standard base names or logic in get_model handles it.
                # If name contains 'hypercolumn', we might need special class or just load weights into densenet?
                # Usually hypercolumn models have specific architecture.
                # If get_model fails, we skip.
                
                # Check if it's a known architecture base
                base_arch = 'densenet169' if 'densenet' in name else name
                model = get_model(base_arch, self.num_classes, pretrained=False).to(self.device)
                
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                self.models[name] = model
                print(f"  ✅ Successfully loaded {name}.")
            except Exception as e:
                print(f"  ❌ Failed to load {name}. Error: {e}. Skipping.")
        
        if not self.models:
            raise RuntimeError("No models were successfully loaded.")

    def _is_hypercolumn_model(self, model_name: str) -> bool:
        """Check if a model is a hypercolumn/column model."""
        return "hypercolumn" in model_name.lower() or "cbam" in model_name.lower()
    
    def _get_weighted_average(self, all_probs: List[np.ndarray], model_names: List[str], 
                               use_hypercolumn_priority: bool) -> np.ndarray:
        """
        Compute weighted average of probabilities.
        """
        weights = []
        for name in model_names:
            if use_hypercolumn_priority and self._is_hypercolumn_model(name):
                weights.append(self.HYPERCOLUMN_WEIGHT)
            else:
                weights.append(self.DEFAULT_WEIGHT)
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
             weights = np.ones(len(weights)) / len(weights)
        
        # Compute weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights):
            weighted_probs += prob * weight
        
        return weighted_probs

    def _predict_with_stacker(self, all_probs: List[np.ndarray], model_names: List[str]):
        """If a `stacker` is present on the instance, use it to predict class probabilities."""
        if not hasattr(self, 'stacker') or self.stacker is None:
            raise RuntimeError('No stacker available')
        
        probs = np.stack(all_probs, axis=0)  # (M, C)
        feat = probs.reshape(1, -1)
        proba = self.stacker.predict_proba(feat)[0]
        return proba
    
    @torch.no_grad()
    def run_ensemble(self, image_input: Union[str, Image.Image], use_stacking: bool = False) -> Dict[str, Any]:
        """Runs ensemble inference on an image (path or object)."""
        if not self.models:
            return {"error": "No models loaded"}
        
        image_path_str = "in-memory-image"
        
        # 1. Image Loading
        if isinstance(image_input, str):
            full_image_path = os.path.abspath(image_input)
            if not os.path.exists(full_image_path):
                return {"error": f"Image file not found at {image_input}"}
            try:
                img = Image.open(full_image_path).convert('RGB')
                image_path_str = image_input
            except Exception as e:
                return {"error": f"Failed to open image at {full_image_path}: {e}"}
        elif isinstance(image_input, Image.Image):
             img = image_input.convert('RGB')
        else:
             return {"error": "Invalid input type. Expected str (path) or PIL.Image."}
        
        input_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        all_probs = []
        model_names = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            model_names.append(name)
            
            pred_idx = np.argmax(probs)
            pred_class_raw = self.class_names[pred_idx]
            
            individual_predictions[name] = {
                "class": _swap_prediction_label(pred_class_raw),
                "confidence": float(probs[pred_idx])
            }
        
        # First pass: compute equal-weighted average to determine likely class
        equal_avg_probs = np.mean(all_probs, axis=0)
        preliminary_idx = np.argmax(equal_avg_probs)
        preliminary_class = self.class_names[preliminary_idx]
        
        # Check if preliminary class is one where hypercolumn models should have priority
        use_hypercolumn_priority = preliminary_class in self.HYPERCOLUMN_PRIORITY_CLASSES
        
        # Second pass: compute final weighted average based on detected class
        if use_stacking:
            try:
                avg_probs = self._predict_with_stacker(all_probs, model_names)
            except Exception:
                avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
        else:
            avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
            
        ensemble_idx = np.argmax(avg_probs)
        ensemble_class_raw = self.class_names[ensemble_idx]
        ensemble_class = _swap_prediction_label(ensemble_class_raw)
        ensemble_confidence = float(avg_probs[ensemble_idx])
        
        # Prepare all probabilities with swapped labels
        all_probs_dict = {}
        for i in range(len(avg_probs)):
            class_name = self.class_names[i]
            swapped_name = _swap_prediction_label(class_name)
            all_probs_dict[swapped_name] = float(avg_probs[i])
            
        # Swap logic for list output
        # (Same as in DiagnosticModule)
        probs_np = avg_probs.copy()
        try:
             if "Transverse" in self.class_names and "Transverse Displaced" in self.class_names:
                idx_trans = self.class_names.index("Transverse")
                idx_trans_disp = self.class_names.index("Transverse Displaced")
                probs_np[idx_trans], probs_np[idx_trans_disp] = probs_np[idx_trans_disp], probs_np[idx_trans]

             if "Oblique" in self.class_names and "Oblique Displaced" in self.class_names:
                idx_obl = self.class_names.index("Oblique")
                idx_obl_disp = self.class_names.index("Oblique Displaced")
                probs_np[idx_obl], probs_np[idx_obl_disp] = probs_np[idx_obl_disp], probs_np[idx_obl]
        except ValueError:
            pass
        
        result = {
            "image_path": image_path_str,
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": probs_np.tolist(),
            "all_probabilities_dict": all_probs_dict,
            "weighted_voting": use_hypercolumn_priority,
            "weighting_reason": f"Hypercolumn models prioritized for {preliminary_class}" if use_hypercolumn_priority else "Equal weights for all models",
            "is_label_swapped": True
        }

        if self.conformal_threshold is not None:
            try:
                conformal_set = predict_conformal_set(avg_probs, self.conformal_threshold, self.class_names)
                result["conformal_set"] = conformal_set
                result["conformal_threshold"] = float(self.conformal_threshold)
            except Exception:
                result["conformal_set_error"] = "failed to compute conformal set"

        return result

# ----------------------------------------------------------------------
# --- Execution Block ---
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Module.')
    parser.add_argument('--image-path', required=True, help='Path to the image.')
    parser.add_argument('--checkpoints-dir', required=True, help='Path to checkpoints directory.')
    parser.add_argument('--models', type=str, default='swin,mobilenetv2,efficientnetv2,maxvit,densenet169', 
                        help='Comma-separated names of the models to load.')
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--class-names', required=True, help='Comma-separated class names.')
    
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(',')]
    class_names_list = [c.strip() for c in args.class_names.split(',')]
    
    try:
        module = EnsembleModule(
            model_names=models_list,
            checkpoints_dir=args.checkpoints_dir,
            num_classes=args.num_classes,
            class_names=class_names_list
        )
    except RuntimeError as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        exit(1)

    result = module.run_ensemble(args.image_path)
    
    print("\n--- ENSEMBLE MODULE RESULT ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['ensemble_prediction']} (Conf: {result['ensemble_confidence']:.4f})")
    print("-----------------------------\n")
