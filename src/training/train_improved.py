"""
Improved Fracture Classification Training Pipeline

Improvements over standard pipeline:
1. Weighted Cross Entropy Loss: Handles class imbalance (specifically for Displaced fractures).
2. Label Smoothing: Prevents overfitting to noisy labels (smoothing=0.1).
3. Mixup Augmentation: Improves generalization and robustness.
4. Test Time Augmentation (TTA): Used during final evaluation for better predictions.

Usage:
    python src/training/train_improved.py \
        --train-csv data/balanced_augmented_dataset/train.csv \
        --val-csv data/balanced_augmented_dataset/val.csv \
        --test-csv data/balanced_augmented_dataset/test.csv \
        --model hypercolumn_densenet169 --epochs 20 --batch-size 8 \
        --out-dir outputs/hypercolumn_improved
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import get_device, get_model, get_transforms, FractureDataset

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ----------------------------- Mixup -----------------------------
def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------- TTA -----------------------------
def tta_predict(model, img_tensor, device):
    """
    Simple TTA: Average prediction of original and horizontal flip.
    img_tensor: (B, C, H, W)
    """
    # Original
    out1 = model(img_tensor).softmax(dim=1)
    
    # Horizontal Flip
    img_flip = torch.flip(img_tensor, dims=[3])
    out2 = model(img_flip).softmax(dim=1)
    
    return (out1 + out2) / 2.0

# ----------------------------- Training -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=0.4, device=device)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        
        # For metrics, we just take the argmax of the output (approximation for mixup)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, p, r, f1

def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            if use_tta:
                probs = tta_predict(model, imgs, device)
                loss = criterion(torch.log(probs + 1e-8), labels) # Approx loss for TTA
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                probs = outputs.softmax(dim=1)
            
            running_loss += loss.item() * imgs.size(0)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return epoch_loss, p, r, f1, cm

def calculate_class_weights(rows, num_classes):
    labels = [int(r['label']) for r in rows]
    counts = Counter(labels)
    total = len(labels)
    weights = []
    print("Class distribution:")
    for i in range(num_classes):
        count = counts.get(i, 0)
        # Inverse frequency weight: N / (C * N_c)
        w = total / (num_classes * count) if count > 0 else 1.0
        weights.append(w)
        print(f"  Class {i}: {count} samples, Weight: {w:.4f}")
    return torch.FloatTensor(weights)

def load_csv_like(path):
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--img-root', default='.')
    parser.add_argument('--model', default='hypercolumn_densenet169')
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out-dir', default='outputs/improved')
    parser.add_argument('--wandb-project', default='fracture-improved')
    parser.add_argument('--wandb-mode', default='disabled')
    
    args = parser.parse_args()
    
    # Init WandB
    if args.wandb_mode != 'disabled':
        wandb.init(project=args.wandb_project, config=args, mode=args.wandb_mode)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load Data
    train_rows = load_csv_like(args.train_csv)
    val_rows = load_csv_like(args.val_csv)
    test_rows = load_csv_like(args.test_csv)
    
    train_ds = FractureDataset(train_rows, img_root=args.img_root, transform=get_transforms('train'))
    val_ds = FractureDataset(val_rows, img_root=args.img_root, transform=get_transforms('val'))
    test_ds = FractureDataset(test_rows, img_root=args.img_root, transform=get_transforms('val'))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, args.num_classes, pretrained=True).to(DEVICE)
    
    # Loss with Class Weights (No Label Smoothing for v2)
    class_weights = calculate_class_weights(train_rows, args.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_f1 = 0.0
    
    print("Starting training (v2: No Mixup, No Label Smoothing)...")
    for epoch in range(args.epochs):
        start = time.time()
        
        # Train without Mixup
        train_loss, train_p, train_r, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, use_mixup=False
        )
        
        # Validate (No TTA for speed)
        val_loss, val_p, val_r, val_f1, cm = validate(
            model, val_loader, criterion, DEVICE, use_tta=False
        )
        
        scheduler.step()
        
        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1
            }, os.path.join(args.out_dir, 'best_improved.pth'))
            print(f"  ★ New Best F1: {val_f1:.4f}")
            
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | Time: {time.time()-start:.1f}s")
        
        if args.wandb_mode != 'disabled':
            wandb.log({
                'train_loss': train_loss, 'train_f1': train_f1,
                'val_loss': val_loss, 'val_f1': val_f1,
                'epoch': epoch
            })

    # Final Evaluation with TTA
    print("\nRunning Final Evaluation with TTA...")
    best_ck = torch.load(os.path.join(args.out_dir, 'best_improved.pth'), map_location=DEVICE)
    model.load_state_dict(best_ck['model_state_dict'])
    
    test_loss, test_p, test_r, test_f1, test_cm = validate(
        model, test_loader, criterion, DEVICE, use_tta=True
    )
    
    print(f"Final Test Results (with TTA):")
    print(f"Macro F1: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    
    if args.wandb_mode != 'disabled':
        wandb.finish()

if __name__ == "__main__":
    main()
