"""
Test Weighted Ensemble: 4 Models (maxvit, hypercolumn_focal, yolo, rad_dino)
Using Weighted Average of Probability Distributions
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add project root and src to path
project_root = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from medai.modules.ensemble_module import EnsembleModule

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    dataset_test = os.path.join(project_root, "balanced_augmented_dataset", "test")
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
    ]
    
    model_names = ["maxvit", "hypercolumn_cbam_densenet169_focal", "yolo", "rad_dino"]
    
    # Define weights for each model
    model_weights = {
        "maxvit": 0.30,
        "hypercolumn_cbam_densenet169_focal": 0.30,
        "yolo": 0.25,
        "rad_dino": 0.15
    }
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    normalized_weights = {k: v / total_weight for k, v in model_weights.items()}
    
    print("=" * 80)
    print("WEIGHTED ENSEMBLE TEST - 4 MODELS (PROBABILITY WEIGHTED)")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"\nModel Weights (Normalized):")
    for name in model_names:
        print(f"  - {name}: {normalized_weights.get(name, 0):.4f}")
    print(f"\nMethod: Weighted Average of Probability Distributions")
    print(f"Device: {device}")
    print()
    
    try:
        # Initialize Ensemble with all 4 models
        print("Initializing Ensemble with 4 models...")
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
        print("✅ Ensemble initialized with all models\n")

    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Monkey-patch to access probabilities directly
    y_true = []
    y_pred = []
    
    print(f"Running weighted ensemble evaluation...")
    print(f"Dataset: {dataset_test}\n")
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            print(f"⚠️ Warning: Directory for {cls_name} not found.")
            continue
                
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Evaluating {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_name)
            
            try:
                # Load image
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    continue
                
                # Get input tensor
                input_tensor = ensemble.transforms(img).unsqueeze(0).to(ensemble.device)
                
                # Collect probabilities from all models
                all_probs = []
                valid_models = []
                
                for name, model in ensemble.models.items():
                    try:
                        if name.lower() == "yolo":
                            # YOLO has its own preprocessing
                            probs = model.predict_pil(img)
                        elif "rad_dino" in name.lower():
                            # RAD-DINO uses HuggingFace processor
                            from medai.modules.ensemble_module import get_rad_dino_input_tensor
                            rad_tensor = get_rad_dino_input_tensor(img, ensemble.device)
                            with torch.no_grad():
                                logits = model(rad_tensor)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        else:
                            # Standard timm model
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        
                        all_probs.append(probs)
                        valid_models.append(name)
                    except Exception as e:
                        pass
                
                if len(all_probs) == 0:
                    continue
                
                # Weighted average of probabilities
                weighted_probs = np.zeros(len(class_names))
                for probs, model_name in zip(all_probs, valid_models):
                    weight = normalized_weights.get(model_name, 1.0 / len(model_names))
                    weighted_probs += probs * weight
                
                # Normalize
                if weighted_probs.sum() > 0:
                    weighted_probs /= len(all_probs)  # Normalize by number of models
                
                # Get prediction
                combined_pred_idx = np.argmax(weighted_probs)
                pred_class_name = class_names[combined_pred_idx]
                
                # Apply label un-swapping for evaluation
                if pred_class_name == "Oblique_Displaced":
                    pred_class_name = "Oblique"
                elif pred_class_name == "Oblique":
                    pred_class_name = "Oblique_Displaced"
                elif pred_class_name == "Transverse_Displaced":
                    pred_class_name = "Transverse"
                elif pred_class_name == "Transverse":
                    pred_class_name = "Transverse_Displaced"
                
                try:
                    pred_idx = class_names.index(pred_class_name)
                    y_true.append(cls_idx)
                    y_pred.append(pred_idx)
                except ValueError:
                    pass
                    
            except Exception as e:
                pass

    # Print Report
    if y_true:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy = (y_true == y_pred).mean()
        
        print("\n" + "=" * 80)
        print("WEIGHTED ENSEMBLE TEST RESULTS")
        print("=" * 80)
        print(f"Models: {', '.join(model_names)}")
        print(f"Method: Weighted Average of Probability Distributions")
        print(f"Total Samples Evaluated: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(y_true))}/{len(y_true)})")
        print()
        print("-" * 80)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("=" * 80)
        
    else:
        print("❌ No predictions were made.")

if __name__ == "__main__":
    main()
