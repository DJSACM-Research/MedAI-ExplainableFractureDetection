import os
import sys
import argparse
import time
import copy
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels
import timm

import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2
import csv 

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import get_device, get_model, get_transforms, FractureDataset

# ----------------------------- Device Selection -----------------------------

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ----------------------------- Training & Evaluation -----------------------------
# (Omitted for brevity, but stays the same as before)
def save_checkpoint(state, is_best, out_dir, name='checkpoint.pth', upload_to_wandb: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, 'best.pth')
        torch.save(state, best_path)
        if upload_to_wandb:
            try:
                wandb.save(best_path)
                print('Uploaded best checkpoint to WandB:', best_path)
            except Exception as e:
                print('WandB save failed:', e)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels, _ in loader:
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
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, p, r, f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', labels=list(range(outputs.shape[1])), zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(outputs.shape[1])))
    return epoch_loss, p, r, f1, cm

# ----------------------------- Helpers: CSV loader -----------------------------
# (Omitted for brevity, but stays the same as before)
def load_csv_like(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf8') as f: 
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# ----------------------------- Main -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, help='train csv', required=True)
    parser.add_argument('--val-csv', type=str, help='val csv', required=True)
    parser.add_argument('--test-csv', type=str, help='test csv', required=True)
    parser.add_argument('--img-root', type=str, default='.', help='root for images')
    parser.add_argument('--model', type=str, default='swin', choices=['swin','convnext','densenet'])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--out-dir', type=str, default='outputs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--stage2', action='store_true', help='run stage 2: generate crops from gradcam and retrain')
    parser.add_argument('--stage2-crop-dir', type=str, default='crops')
    parser.add_argument('--cam-layer', type=str, default=None, help='module name for Grad-CAM hook (optional)')

    # wandb args
    parser.add_argument('--wandb-project', type=str, default='fracture-mps')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online','offline','disabled'])

    args = parser.parse_args(argv)

    if args.wandb_mode != 'disabled':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, mode=args.wandb_mode)
        wandb.config.update(vars(args))
    else:
        wandb.init(mode='disabled') 

    device = DEVICE 

    train_rows = load_csv_like(args.train_csv)
    val_rows = load_csv_like(args.val_csv)
    test_rows = load_csv_like(args.test_csv)

    train_tf = get_transforms('train', img_size=args.img_size)
    val_tf = get_transforms('val', img_size=args.img_size)

    model = get_model(args.model, args.num_classes, pretrained=True).to(device) 

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = ck.get('model_state_dict', ck) 
        model.load_state_dict(state_dict)
        print('Loaded checkpoint', args.checkpoint)

    pin_memory = device.type == 'cuda' 
    num_workers = 0 if device.type == 'cuda' else 4 

    train_ds = FractureDataset(train_rows, img_root=args.img_root, transform=train_tf)
    val_ds = FractureDataset(val_rows, img_root=args.img_root, transform=val_tf)
    test_ds = FractureDataset(test_rows, img_root=args.img_root, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    # FIX: Corrected typo from args.batch-size to args.batch_size
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))

    best_f1 = 0.0
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_p, train_r, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_p, val_r, val_f1, cm = validate(model, val_loader, criterion, device)
        scheduler.step()
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
        ck_name = f'epoch_{epoch}.pth'
        
        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_f1': val_f1}, is_best, out_dir, name=ck_name, upload_to_wandb=(args.wandb_mode!='disabled'))

        # wandb logging
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'train_macro_f1': train_f1, 'val_loss': val_loss, 'val_macro_f1': val_f1, 'lr': scheduler.get_last_lr()[0]}
        print(f"Epoch {epoch}/{args.epochs} time={time.time()-start:.1f}s")
        print(metrics)
        if args.wandb_mode != 'disabled':
            wandb.log(metrics, step=epoch)
            # log confusion matrix as an image
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(cm, interpolation='nearest')
                ax.set_title('Confusion matrix')
                wandb.log({"confusion_matrix": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
            except Exception as e:
                print('Failed to log confusion matrix plot to wandb:', e)

    # load best and final test evaluation
    best_ck = os.path.join(out_dir, 'best.pth')
    if os.path.exists(best_ck):
        ck = torch.load(best_ck, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded best checkpoint for final evaluation')

    test_loss, test_p, test_r, test_f1, test_cm = validate(model, test_loader, criterion, device)
    print('Test results:', test_loss, test_p, test_r, test_f1)
    np.savetxt(os.path.join(out_dir, 'confusion_matrix.txt'), test_cm, fmt='%d')

    if args.wandb_mode != 'disabled':
        try:
            wandb.log({'test_macro_f1': test_f1})
            wandb.save(os.path.join(out_dir, 'confusion_matrix.txt'))
        except Exception as e:
            print('WandB final save failed:', e)

    print('Finished.')
    if args.wandb_mode != 'disabled':
        wandb.finish()

if __name__ == '__main__':
    main()