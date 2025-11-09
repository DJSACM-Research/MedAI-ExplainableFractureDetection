import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels
import timm

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2
import csv
import matplotlib.pyplot as plt

# Import necessary modules for Grad-CAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import get_device, get_model, get_transforms

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ----------------------------- Dataset (Reusing logic from pipeline.py) -----------------------------

class FractureDataset(Dataset):
    def __init__(self, df, img_root: str = '.', transform=None):
        self.entries = df
        self.img_root = img_root
        self.transform = transform
        # CRITICAL PATH FIX: Define the redundant prefix
        self.redundant_prefix = 'balanced_augmented_dataset/' 
        self.redundant_prefix_len = len(self.redundant_prefix)

    def __len__(self):
        return len(len(self.entries))

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = row['image_path']
        
        # PATH CLEANING FIX: Strip the redundant prefix
        if img_path.startswith(self.redundant_prefix):
            img_path = img_path[self.redundant_prefix_len:]

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)
            
        img = Image.open(img_path).convert('RGB')
        
        # NOTE: We return the raw image here for visualization purposes
        raw_img = np.array(img).astype(np.float32) / 255.0

        label = int(row['label']) 
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_path, raw_img


# ----------------------------- Model selection with Grad-CAM target layers -----------------------------

def get_model_with_target_layer(name: str, num_classes: int, pretrained: bool=True):
    """Get model and its target layer for Grad-CAM visualization."""
    model = get_model(name, num_classes, pretrained=pretrained)
    name = name.lower()
    
    if name.startswith('swin'):
        # Target layer for Swin: the last layer of the last stage (blocks[-1][-1])
        target_layer = model.layers[-1].blocks[-1].norm2
        return model, target_layer
    
    if name.startswith('convnext'):
        # Target layer for ConvNext: the last block of the feature extractor
        target_layer = model.stages[-1]
        return model, target_layer
    
    if name.startswith('densenet'):
        # Target layer for DenseNet: features.norm5
        target_layer = model.features.norm5
        return model, target_layer

    raise ValueError(f'Unknown target layer for model: {name}')


# ----------------------------- Helpers: CSV loader -----------------------------

def load_csv_like(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf8') as f: 
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# ----------------------------- Grad-CAM Analysis -----------------------------

def analyze(args):
    device = DEVICE

    # Load CSVs
    test_rows = load_csv_like(args.test_csv)
    
    # Get model and the target layer for Grad-CAM
    model, target_layer = get_model_with_target_layer(args.model, args.num_classes, pretrained=False)
    model.to(device)

    # Load checkpoint weights
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f'Loaded model from {args.checkpoint} onto {device}.')

    # Data setup
    test_tf = get_transforms('val', args.img_size)
    test_ds = FractureDataset(test_rows, img_root=args.img_root, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Use batch size 1 for accurate CAM per image

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))

    # Setup output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    class_names = args.class_names.split(',')
    
    print(f"Starting Grad-CAM analysis on {len(test_ds)} images...")

    for i, (imgs, labels, img_paths, raw_imgs) in enumerate(test_loader):
        imgs = imgs.to(device)
        true_label = labels.item()
        
        # 1. Prediction and Target Setup
        with torch.no_grad():
            outputs = model(imgs)
            predicted_label = outputs.softmax(dim=1).argmax(dim=1).item()

        # Set the target to the PREDICTED class for visualization
        targets = [ClassifierOutputTarget(predicted_label)]
        
        # 2. Generate CAM
        grayscale_cam = cam(input_tensor=imgs, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 3. Visualization
        # raw_img is the unnormalized image [0, 1]
        raw_img_for_viz = raw_imgs.squeeze(0).numpy()
        visualization = show_cam_on_image(raw_img_for_viz, grayscale_cam, use_rgb=True)
        
        # Convert to PIL Image for saving
        visualization_pil = Image.fromarray(cv2.cvtColor((visualization * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # 4. Save
        path_obj = Path(img_paths[0])
        class_name = class_names[true_label]
        
        # Define saving path
        save_dir = os.path.join(args.out_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine the name with prediction/truth info
        pred_class_name = class_names[predicted_label]
        file_name = f'CAM_T{class_name}_P{pred_class_name}_{path_obj.name}'
        save_path = os.path.join(save_dir, file_name)
        
        visualization_pil.save(save_path)
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(test_ds)}. Saved to: {save_path}")

    print("Grad-CAM analysis complete. Results saved to:", args.out_dir)


# ----------------------------- Main -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Grad-CAM analysis on test data.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (e.g., outputs/swin_mps/best.pth)')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to the test CSV file.')
    parser.add_argument('--img-root', type=str, default='.', help='Root directory for images.')
    parser.add_argument('--model', type=str, default='swin', choices=['swin','convnext'])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--out-dir', type=str, default='outputs/analysis', help='Directory to save CAM visualizations.')
    parser.add_argument('--class-names', type=str, required=True, 
                        help='Comma-separated list of class names (e.g., "A,B,C")')
    
    args = parser.parse_args()
    
    # Check for required library dependencies
    try:
        import pytorch_grad_cam
    except ImportError:
        print("ERROR: pytorch-grad-cam library not found. Please install it:")
        print("pip install pytorch-grad-cam")
        exit(1)

    analyze(args)