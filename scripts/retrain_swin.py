#!/usr/bin/env python3
"""
Retrain only the Swin model with the corrected dataset.
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

from src.utils import get_model, get_transforms, FractureDataset

# Configuration
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_CLASSES = 8
DATA_ROOT = 'data'
IMG_SIZE = 224


def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
    print("RETRAINING SWIN MODEL")
    print("="*60)
    
    # Create model
    model = get_model('swin', num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Load data
    train_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/train.csv').to_dict('records')
    val_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/val.csv').to_dict('records')
    
    train_transform = get_transforms('train', IMG_SIZE)
    val_transform = get_transforms('test', IMG_SIZE)
    
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc / 100.0,
                'epoch': epoch + 1,
                'model_name': 'swin',
            }, 'models/best_swin.pth')
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"\nSWIN Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on TEST SET")
    print("="*60)
    
    checkpoint = torch.load('models/best_swin.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/test.csv').to_dict('records')
    test_transform = get_transforms('test', IMG_SIZE)
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
    print("\n✅ Swin model retrained successfully!")


if __name__ == '__main__':
    main()
