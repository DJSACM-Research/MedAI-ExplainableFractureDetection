"""
Test Combo 3: maxvit, yolo, rad_dino
Using Stacking Ensemble
"""
import os
import sys
import torch
import numpy as np
import joblib
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
    stacker_path = os.path.join(project_root, "outputs", "stacker.joblib")
    dataset_test = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\balanced_augmented_dataset\test"
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
    ]
    
    # Test Combo 3: maxvit, yolo, rad_dino
    model_names = ["maxvit", "yolo", "rad_dino"]

    print("="*70)
    print("TEST COMBO 3: maxvit, yolo, rad_dino")
    print("="*70)
    print("Initializing Ensemble...")
    
    try:
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
        
        if os.path.exists(stacker_path):
            ensemble.stacker = joblib.load(stacker_path)
            print(f"✅ Loaded stacker from {stacker_path}")
            print("⚠️ Note: Stacker was trained with 4 models, using 3 models may affect results")
        else:
            print(f"❌ Stacker not found. Using weighted average instead.")
            ensemble.stacker = None

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # Run Evaluation
    y_true = []
    y_pred = []
    
    print(f"\nRunning evaluation on {dataset_test}...")
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            cls_dir = os.path.join(dataset_test, cls_name.replace(" ", "_"))
            if not os.path.exists(cls_dir):
                print(f"⚠️ Warning: Directory for {cls_name} not found.")
                continue
                
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Evaluating {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_name)
            
            try:
                result = ensemble.run_ensemble(img_path, use_stacking=True)
                
                if "error" in result:
                    continue
                
                pred_class = result["ensemble_prediction"]
                pred_class_unswapped = pred_class.replace(" ", "_")
                
                # Un-swap labels for evaluation
                if pred_class_unswapped == "Oblique_Displaced":
                    pred_class_unswapped = "Oblique"
                elif pred_class_unswapped == "Oblique":
                    pred_class_unswapped = "Oblique_Displaced"
                elif pred_class_unswapped == "Transverse_Displaced":
                    pred_class_unswapped = "Transverse"
                elif pred_class_unswapped == "Transverse":
                    pred_class_unswapped = "Transverse_Displaced"
                
                try:
                    pred_idx = class_names.index(pred_class_unswapped)
                    y_true.append(cls_idx)
                    y_pred.append(pred_idx)
                except ValueError:
                    pass
                    
            except Exception as e:
                pass

    # Print Report
    if y_true:
        print("\n" + "="*70)
        print("TEST COMBO 3 RESULTS - STACKING ENSEMBLE EVALUATION")
        print("="*70)
        print(f"Models: {', '.join(model_names)}")
        print(f"Total Samples Evaluated: {len(y_true)}")
        print("\n" + "-"*70)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*70)
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    main()
