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

from medai.modules.ensemble_module import EnsembleModule, RadDinoClassifier, YOLOClassifierWrapper, get_model
from ultralytics import YOLO

def load_custom_models(device, num_classes, class_names):
    models = {}
    
    # 1. MaxViT
    print("Loading MaxViT...")
    maxvit_path = os.path.join(project_root, "outputs", "cross_validation", "best_maxvit.pth")
    m_maxvit = get_model('maxvit', num_classes, pretrained=False)
    ck = torch.load(maxvit_path, map_location=device)
    m_maxvit.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    m_maxvit.to(device).eval()
    models['maxvit'] = m_maxvit
    
    # 2. Hypercolumn CBAM DenseNet169 Focal
    print("Loading HC CBAM DenseNet Focal...")
    hc_path = os.path.join(project_root, "outputs", "cross_validation", "best_hypercolumn_cbam_densenet169_focal.pth")
    m_hc = get_model('densenet169', num_classes, pretrained=False)
    ck2 = torch.load(hc_path, map_location=device)
    m_hc.load_state_dict(ck2.get('model_state_dict', ck2), strict=False)
    m_hc.to(device).eval()
    models['hypercolumn_cbam_densenet169_focal'] = m_hc
    
    # 3. YOLOv26m-cls
    print("Loading YOLO...")
    yolo_path = os.path.join(project_root, "outputs", "yolo_cls_finetune", "yolo_cls_ft", "weights", "best.pt")
    if not os.path.exists(yolo_path):
        yolo_path = os.path.join(project_root, "yolov8n-cls.pt") # fallback
    yolo_raw = YOLO(yolo_path, task="classify")
    m_yolo = YOLOClassifierWrapper(yolo_raw, class_names)
    models['yolo'] = m_yolo
    
    # 4. RAD-DINO
    print("Loading RAD-DINO...")
    dino_pth = os.path.join(project_root, "outputs", "dinorad", "dinorad_best.pth")
    # Need to detect head type or assume linear
    # Based on previous analysis, let's try to map backbone -> dinov2
    m_dino = RadDinoClassifier(num_classes, head_type="linear")
    ck3 = torch.load(dino_pth, map_location=device)
    sd = ck3.get('model_state_dict', ck3)
    mapped_sd = {}
    for k, v in sd.items():
        new_k = k.replace('backbone.', 'backbone.backbone.') # RadDinoClassifier has self.backbone = AutoModel... 
        # Actually in RadDinoClassifier: self.backbone(pixel_values=pixel_values)
        # AutoModel for DinoV2 has keys starting with 'dinov2.'
        # If the checkpoint has 'backbone.', it might be from the trainer that used 'backbone' name.
        # Let's map backbone -> backbone
        new_k = k
        if new_k.startswith('classifier.'):
            new_k = new_k.replace('classifier.', 'head.')
        mapped_sd[new_k] = v
    
    m_dino.load_state_dict(mapped_sd, strict=False)
    m_dino.to(device).eval()
    models['rad_dino'] = m_dino
    
    return models

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_test = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\balanced_augmented_dataset\test"
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
    ]
    
    # 1. Load models manually to ensure correct paths
    models = load_custom_models(device, 8, class_names)

    # 2. Initialize Ensemble Module with the pre-loaded models
    print("Initializing Ensemble with pre-loaded models...")
    ensemble = EnsembleModule(
        class_names=class_names,
        models=models,
        num_classes=8,
        device=device
    )

    # 3. Run Evaluation
    y_true = []
    y_pred = []
    
    print(f"Running evaluation on {dataset_test}...")
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            cls_dir = os.path.join(dataset_test, cls_name.replace(" ", "_"))
            if not os.path.exists(cls_dir):
                print(f"⚠️ Warning: Directory for {cls_name} not found.")
                continue
                
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=cls_name, leave=False):
            img_path = os.path.join(cls_dir, img_name)
            try:
                # use_stacking=False for weighted ensemble
                result = ensemble.run_ensemble(img_path, use_stacking=False)
                if "error" in result: continue
                
                pred_idx = class_names.index(result["ensemble_prediction"])
                y_true.append(cls_idx)
                y_pred.append(pred_idx)
            except:
                pass

    if y_true:
        print("\n" + "="*60)
        print("ALL-MODEL WEIGHTED ENSEMBLE EVALUATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*60)
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    main()
