"""
Test Weighted Ensemble - 4 Models: maxvit, hypercolumn_focal, yolo, rad_dino
No stacking - using optimized weights for probability averaging
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

project_root = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from medai.modules.ensemble_module import EnsembleModule

def get_model_predictions(ensemble, img_path, img, class_names):
    """Get probability predictions from all models"""
    try:
        input_tensor = ensemble.transforms(img).unsqueeze(0).to(ensemble.device)
        all_probs = []
        valid_models = []
        
        for name, model in ensemble.models.items():
            try:
                if name.lower() == "yolo":
                    probs = model.predict_pil(img)
                elif "rad_dino" in name.lower():
                    from medai.modules.ensemble_module import get_rad_dino_input_tensor
                    rad_tensor = get_rad_dino_input_tensor(img, ensemble.device)
                    with torch.no_grad():
                        logits = model(rad_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                all_probs.append(probs)
                valid_models.append(name)
            except:
                pass
        
        return all_probs, valid_models
    except:
        return [], []

def weighted_ensemble_inference(all_probs, valid_models, model_weights, class_names):
    """Combine probabilities using custom weights"""
    if len(all_probs) == 0:
        return None, None
    
    total_weight = 0.0
    weighted_probs = np.zeros(len(class_names))
    
    for probs, model_name in zip(all_probs, valid_models):
        weight = model_weights.get(model_name, 1.0 / len(valid_models))
        weighted_probs += probs * weight
        total_weight += weight
    
    # Normalize
    if total_weight > 0:
        weighted_probs /= total_weight
    
    pred_idx = np.argmax(weighted_probs)
    return pred_idx, weighted_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    dataset_test = os.path.join(project_root, "balanced_augmented_dataset", "test")
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
    ]
    
    model_names = ["maxvit", "hypercolumn_cbam_densenet169_focal", "yolo", "rad_dino"]
    
    # Optimized weights based on individual model performance
    model_weights = {
        "maxvit": 0.25,
        "hypercolumn_cbam_densenet169_focal": 0.35,
        "yolo": 0.25,
        "rad_dino": 0.15
    }
    
    print("=" * 80)
    print("WEIGHTED ENSEMBLE - 4 MODELS (PROBABILITY AVERAGING)")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"\nModel Weights:")
    for name in model_names:
        print(f"  - {name}: {model_weights.get(name, 0):.2f}")
    print(f"\nTotal Weight: {sum(model_weights.values()):.2f}")
    print(f"Device: {device}\n")
    
    try:
        print("Loading ensemble...")
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
        print("✅ Ensemble loaded\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    y_true = []
    y_pred = []
    
    print("Evaluating on test set...\n")
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            continue
        
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Evaluating {cls_name}", leave=False):
            try:
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                
                all_probs, valid_models = get_model_predictions(ensemble, img_path, img, class_names)
                if len(all_probs) == 0:
                    continue
                
                pred_idx, _ = weighted_ensemble_inference(all_probs, valid_models, model_weights, class_names)
                
                if pred_idx is None:
                    continue
                
                pred_class_name = class_names[pred_idx]
                
                # Un-swap labels for evaluation
                if pred_class_name == "Oblique_Displaced":
                    pred_class_name = "Oblique"
                elif pred_class_name == "Oblique":
                    pred_class_name = "Oblique_Displaced"
                elif pred_class_name == "Transverse_Displaced":
                    pred_class_name = "Transverse"
                elif pred_class_name == "Transverse":
                    pred_class_name = "Transverse_Displaced"
                
                pred_idx = class_names.index(pred_class_name)
                y_true.append(cls_idx)
                y_pred.append(pred_idx)
            except:
                pass
    
    # Results
    if y_true:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        print("\n" + "=" * 80)
        print("RESULTS - WEIGHTED ENSEMBLE (4 MODELS)")
        print("=" * 80)
        print(f"Total Samples: {len(y_true)}")
        print(f"Accuracy: {acc:.4f}")
        print("-" * 80)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("=" * 80)

if __name__ == "__main__":
    main()
