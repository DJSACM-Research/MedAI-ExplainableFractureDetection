#!/usr/bin/env python3
"""
Retrain the HyperColumn-CBAM and Swin models with the corrected dataset.

These models were trained before the test label fix and need to be retrained
with the correct label mappings.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm
import os

from src.utils import get_model, get_transforms, FractureDataset
from src.models import HyperColumnCBAMDenseNet169

# Configuration
MODELS_TO_RETRAIN = {
    'hypercolumn_cbam': {
        'model_class': 'hypercolumn_cbam',
        'save_name': 'best_hypercolumn_cbam_densenet169.pth',
        'img_size': 224,
    },
    'swin': {
        'model_class': 'swin',
        'save_name': 'best_swin.pth',
        'img_size': 224,
    },
}

# Training parameters
EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_CLASSES = 8
DATA_ROOT = 'data'


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train_model(model_name, config, device):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create model
    if model_name == 'hypercolumn_cbam':
        model = HyperColumnCBAMDenseNet169(num_classes=NUM_CLASSES, pretrained=True)
    else:
        model = get_model(config['model_class'], num_classes=NUM_CLASSES, pretrained=True)
    
    model = model.to(device)
    
    # Load data
    train_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/train.csv').to_dict('records')
    val_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/val.csv').to_dict('records')
    
    img_size = config['img_size']
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('test', img_size)
    
    train_dataset = FractureDataset(train_df, DATA_ROOT, transform=train_transform)
    val_dataset = FractureDataset(val_df, DATA_ROOT, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # Save best model
            save_path = f"models/{config['save_name']}"
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc / 100.0,
                'epoch': epoch + 1,
                'model_name': model_name,
            }, save_path)
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"\n{model_name.upper()} Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    return best_val_acc


def evaluate_on_test(model_name, config, device):
    """Evaluate retrained model on test set."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} on TEST SET")
    print(f"{'='*60}")
    
    # Load model
    if model_name == 'hypercolumn_cbam':
        model = HyperColumnCBAMDenseNet169(num_classes=NUM_CLASSES, pretrained=False)
    else:
        model = get_model(config['model_class'], num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(f"models/{config['save_name']}", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/test.csv').to_dict('records')
    test_transform = get_transforms('test', config['img_size'])
    test_dataset = FractureDataset(test_df, DATA_ROOT, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_acc


def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("\n" + "="*60)
    print("RETRAINING OLD MODELS (HyperColumn-CBAM & Swin)")
    print("="*60)
    print(f"These models were trained before the test label fix.")
    print(f"Retraining with corrected dataset...")
    
    results = {}
    
    # Train each model
    for model_name, config in MODELS_TO_RETRAIN.items():
        val_acc = train_model(model_name, config, device)
        test_acc = evaluate_on_test(model_name, config, device)
        results[model_name] = {'val_acc': val_acc, 'test_acc': test_acc}
    
    # Summary
    print("\n" + "="*60)
    print("RETRAINING SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"{model_name:25s}: Val {metrics['val_acc']:.2f}% | Test {metrics['test_acc']:.2f}%")
    
    print("\n✅ All models retrained successfully!")


if __name__ == '__main__':
    main()
