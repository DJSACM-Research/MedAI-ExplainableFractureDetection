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
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    dataset_test = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\balanced_augmented_dataset\test"
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
    ]
    
    # Models to include in weighted ensemble
    model_names = ["maxvit", "yolo", "hypercolumn_cbam_densenet169", "rad_dino"]

    # 2. Initialize Ensemble Module
    print("Initializing Weighted Ensemble...")
    try:
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 3. Run Evaluation on Test Set
    y_true = []
    y_pred = []
    
    print(f"Running evaluation on {dataset_test}...")
    
    # Iterate through class folders
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
                # use_stacking=False (default) uses the weighting logic in get_weighted_average
                result = ensemble.run_ensemble(img_path, use_stacking=False)
                
                if "error" in result:
                    continue
                
                pred_class = result["ensemble_prediction"]
                
                try:
                    pred_idx = class_names.index(pred_class)
                    y_true.append(cls_idx)
                    y_pred.append(pred_idx)
                except ValueError:
                    pass
                    
            except Exception as e:
                pass

    # 4. Print Final Report
    if y_true:
        print("\n" + "="*60)
        print("WEIGHTED ENSEMBLE EVALUATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*60)
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    main()
