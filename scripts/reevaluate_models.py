#!/usr/bin/env python3
"""Re-evaluate trained models after fixing test labels."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils import get_model, get_transforms, FractureDataset

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def evaluate_model(model_path, model_name, test_loader, device):
    """Evaluate a model on the test set."""
    # Load model
    model = get_model(model_name, num_classes=8, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return acc, p, r, f1

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Load test data
    test_csv = 'data/balanced_augmented_dataset/test.csv'
    test_df = pd.read_csv(test_csv).to_dict('records')
    test_transform = get_transforms('test', 224)
    test_dataset = FractureDataset(test_df, 'data', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Models to evaluate
    models = [
        ('models/best_densenet169.pth', 'densenet169'),
        ('models/best_efficientnetv2.pth', 'efficientnet'),
        ('models/best_maxvit.pth', 'maxvit'),
        ('models/best_mobilenetv2.pth', 'mobilenet'),
        ('models/best_hypercolumn_cbam_densenet169.pth', 'hypercolumn_densenet169'),
        ('models/best_swin.pth', 'swin'),
    ]
    
    print("\n" + "="*70)
    print("RE-EVALUATION WITH FIXED TEST LABELS")
    print("="*70)
    print(f"\n{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)
    
    results = []
    for model_path, model_name in models:
        if os.path.exists(model_path):
            try:
                acc, p, r, f1 = evaluate_model(model_path, model_name, test_loader, device)
                print(f"{model_name:<35} {acc:<12.4f} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
                results.append((model_name, acc, f1))
            except Exception as e:
                print(f"{model_name:<35} ERROR: {str(e)[:40]}")
        else:
            print(f"{model_name:<35} NOT FOUND")
    
    if results:
        print("\n" + "="*70)
        best = max(results, key=lambda x: x[1])
        print(f"Best model: {best[0]} with {best[1]:.2%} accuracy")

if __name__ == '__main__':
    main()
