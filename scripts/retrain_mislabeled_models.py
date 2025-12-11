#!/usr/bin/env python3
"""
Retrain mislabeled models with their correct architectures.

This script retrains the following models that were incorrectly saved with Swin weights:
- DenseNet169
- EfficientNet-B0  
- MaxViT
- MobileNetV2

Usage:
    python scripts/retrain_mislabeled_models.py [--model MODEL] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
    
Examples:
    # Retrain all 4 models
    python scripts/retrain_mislabeled_models.py
    
    # Retrain only DenseNet169
    python scripts/retrain_mislabeled_models.py --model densenet169
    
    # Retrain with custom epochs
    python scripts/retrain_mislabeled_models.py --epochs 30
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from src.utils import get_model, get_transforms, FractureDataset

# Models to retrain (those that were mislabeled)
MODELS_TO_RETRAIN = {
    'densenet169': {
        'output_name': 'best_densenet169.pth',
        'description': 'DenseNet169 - Dense connections with feature reuse'
    },
    'efficientnet': {
        'output_name': 'best_efficientnetv2.pth', 
        'description': 'EfficientNet-B0 - Compound scaling with mobile inverted bottlenecks'
    },
    'maxvit': {
        'output_name': 'best_maxvit.pth',
        'description': 'MaxViT - Multi-axis attention with conv-transformer hybrid'
    },
    'mobilenet': {
        'output_name': 'best_mobilenetv2.pth',
        'description': 'MobileNetV2 - Lightweight depthwise separable convolutions'
    }
}


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if scheduler is not None:
        scheduler.step()
    
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, acc, p, r, f1


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc='Validating', leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, acc, p, r, f1


def train_model(model_name, config, args):
    """Train a single model."""
    print(f"\n{'='*70}")
    print(f"Training: {model_name.upper()}")
    print(f"Description: {config['description']}")
    print(f"Output: models/{config['output_name']}")
    print(f"{'='*70}\n")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Data paths
    train_csv = 'data/balanced_augmented_dataset/train.csv'
    val_csv = 'data/balanced_augmented_dataset/val.csv'
    test_csv = 'data/balanced_augmented_dataset/test.csv'
    img_root = 'data'
    
    # Check if data exists
    if not os.path.exists(train_csv):
        print(f"ERROR: Training data not found at {train_csv}")
        return None
    
    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name, num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get transforms
    train_transform = get_transforms('train', args.img_size)
    val_transform = get_transforms('val', args.img_size)
    
    # Load CSV files
    train_df = pd.read_csv(train_csv).to_dict('records')
    val_df = pd.read_csv(val_csv).to_dict('records')
    test_df = pd.read_csv(test_csv).to_dict('records')
    
    # Create datasets
    train_dataset = FractureDataset(train_df, img_root, transform=train_transform)
    val_dataset = FractureDataset(val_df, img_root, transform=val_transform)
    test_dataset = FractureDataset(test_df, img_root, transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_p, train_r, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Validate
        val_loss, val_acc, val_p, val_r, val_f1 = validate(model, val_loader, criterion, device)
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_val_acc,
                'model_name': model_name,
                'architecture': model_name,
            }
            print(f"  *** New best model! Val Acc: {best_val_acc:.4f}")
    
    # Save best model to models directory
    if best_model_state is not None:
        output_path = os.path.join('models', config['output_name'])
        torch.save(best_model_state, output_path)
        print(f"\n✓ Saved best model to: {output_path}")
        
        # Also save to outputs directory
        output_dir = os.path.join('outputs', f'retrain_{model_name}')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(output_dir, 'best.pth'))
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    # Load best model
    model.load_state_dict(best_model_state['model_state_dict'])
    test_loss, test_acc, test_p, test_r, test_f1 = validate(model, test_loader, criterion, device)
    
    print(f"Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_p:.4f}")
    print(f"  Recall:    {test_r:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    return {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_precision': test_p,
        'test_recall': test_r,
        'test_f1': test_f1,
        'epochs_trained': args.epochs,
        'output_path': os.path.join('models', config['output_name'])
    }


def main():
    parser = argparse.ArgumentParser(description='Retrain mislabeled models')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODELS_TO_RETRAIN.keys()),
                        help='Specific model to retrain (default: all)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs (default: 25)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--num-classes', type=int, default=8,
                        help='Number of output classes (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RETRAINING MISLABELED MODELS")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    
    # Determine which models to train
    if args.model:
        models_to_train = {args.model: MODELS_TO_RETRAIN[args.model]}
    else:
        models_to_train = MODELS_TO_RETRAIN
    
    print(f"\nModels to retrain: {list(models_to_train.keys())}")
    
    # Train each model
    results = []
    for model_name, config in models_to_train.items():
        try:
            result = train_model(model_name, config, args)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    if results:
        print(f"\n{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
        print("-"*60)
        for r in results:
            print(f"{r['model_name']:<20} {r['best_val_acc']:<12.4f} {r['test_acc']:<12.4f} {r['test_f1']:<12.4f}")
        
        print(f"\nAll models saved to: models/")
    else:
        print("No models were successfully trained.")
    
    return results


if __name__ == '__main__':
    main()
