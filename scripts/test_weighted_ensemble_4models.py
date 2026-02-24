"""
Test Weighted Ensemble: 4 Models (maxvit, hypercolumn_focal, yolo, rad_dino)
Using Custom Weighted Average instead of Stacking
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
    
    # Define weights for each model (will be normalized)
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
    print("WEIGHTED ENSEMBLE TEST - 4 MODELS")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"\nModel Weights (Normalized):")
    for name in model_names:
        print(f"  - {name}: {normalized_weights.get(name, 0):.4f}")
    print(f"\nTotal Weight: {sum(normalized_weights.values()):.4f}")
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

    # Run Evaluation
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
                
                # Run ensemble without stacking to get individual model predictions
                result = ensemble.run_ensemble(img_path, use_stacking=False)
                
                if "error" in result:
                    continue
                
                # Get individual predictions from each model
                individual_preds = result.get("individual_predictions", {})
                
                # Extract predictions and apply weights using probability voting
                # We need to access all_probs during the run, so let's use a modified approach
                
                # For now, we'll use the predicted class from each model for weighted voting
                weighted_vote = np.zeros(len(class_names))
                valid_models = 0
                
                for model_name in model_names:
                    if model_name in individual_preds:
                        pred_info = individual_preds[model_name]
                        pred_class = pred_info["class"]
                        weight = normalized_weights.get(model_name, 1.0 / len(model_names))
                        
                        try:
                            # Normalize class name to use underscores
                            pred_class_normalized = pred_class.replace(" ", "_")
                            pred_idx = class_names.index(pred_class_normalized)
                            weighted_vote[pred_idx] += weight
                            valid_models += 1
                        except ValueError:
                            continue
                
                if valid_models == 0:
                    continue
                
                # Get prediction from weighted voting
                combined_pred_idx = np.argmax(weighted_vote)
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
        
        print("\n" + "=" * 80)
        print("WEIGHTED ENSEMBLE TEST RESULTS")
        print("=" * 80)
        print(f"Models: {', '.join(model_names)}")
        print(f"Method: Weighted Voting (Custom Per-Model Weights)")
        print(f"Total Samples Evaluated: {len(y_true)}")
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
