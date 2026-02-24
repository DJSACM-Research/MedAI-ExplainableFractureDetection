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
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    stacker_path = os.path.join(project_root, "outputs", "stacker.joblib")
    dataset_test = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\balanced_augmented_dataset\test"
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
    ]
    
    # These MUST match the order and names used during stacker training (seen in stacker_eval.json)
    model_names = ["maxvit", "yolo", "hypercolumn_cbam_densenet169", "rad_dino"]

    # 2. Initialize Ensemble Module
    # Note: EnsembleModule expects checkpoints in checkpoints_dir with "best_{name}.pth"
    # Rad-Dino and YOLO have special handling inside the module
    print("Initializing Stacking Ensemble...")
    try:
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
        
        # 3. Load the Meta-Model (Stacker)
        if os.path.exists(stacker_path):
            ensemble.stacker = joblib.load(stacker_path)
            print(f"✅ Loaded stacker from {stacker_path}")
        else:
            print(f"❌ Stacker not found at {stacker_path}. Using weighted average instead.")
            ensemble.stacker = None

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 4. Run Evaluation on Test Set
    y_true = []
    y_pred = []
    
    print(f"Running evaluation on {dataset_test}...")
    
    # Iterate through class folders
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            # Try with underscores if spaces failed (e.g. Oblique_Displaced)
            cls_dir = os.path.join(dataset_test, cls_name.replace(" ", "_"))
            if not os.path.exists(cls_dir):
                print(f"⚠️ Warning: Directory for {cls_name} not found.")
                continue
                
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Evaluating {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_name)
            
            try:
                # use_stacking=True tells the module to use the Logistic Regression model
                result = ensemble.run_ensemble(img_path, use_stacking=True)
                
                if "error" in result:
                    continue
                
                pred_class = result["ensemble_prediction"]
                
                # Map back to index for reporting - normalize class names (spaces → underscores)
                # Un-swap prediction labels to match true labels for evaluation
                pred_class_unswapped = pred_class.replace(" ", "_")
                # Undo the label swaps (reverse the swap function logic)
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
                    print(f"Unknown pred class: {pred_class_unswapped}")
                    
            except Exception as e:
                # print(f"Error pred: {e}")
                pass

    # 5. Print Final Report
    if y_true:
        print("\n" + "="*60)
        print("STACKING ENSEMBLE EVALUATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*60)
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    main()
