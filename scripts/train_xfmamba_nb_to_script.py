#!/usr/bin/env python
# coding: utf-8

# # XFMamba Fracture Classification Training (Pure Python)
# 
# This notebook trains an XFMamba-style vision model using a **Pure PyTorch implementation** of the Mamba block. 
# This avoids complex CUDA compilation issues on Windows while still leveraging the State Space Model (SSM) architecture for long-range dependency modeling.

# In[ ]:


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange, repeat

# Check Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[ ]:


# --- Pure PyTorch Mamba Implementation ---
# Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
# Simplified for training without custom CUDA kernels

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        batch, seq_len, d_model = x.shape
        
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        # Runs the SSM. In pure PyTorch, this is an unrolled loop (slow but functional).
        # x: [B, L, D_inner]
        d_inner = self.d_inner
        batch, seq_len, _ = x.shape
        
        A = -torch.exp(self.A_log.float())  # [D, N]
        D = self.D.float()

        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # [B, L, D_inner]
        
        y = []
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        
        # Scan loop (the slow part in pure python)
        for t in range(seq_len):
            dt = delta[:, t, :] # [B, D]
            Bt = B[:, t, :] # [B, N]
            Ct = C[:, t, :] # [B, N]
            xt = x[:, t, :] # [B, D]
            
            # Discretization
            dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
            dB = torch.einsum('bd,bn->bdn', dt, Bt)
            
            # State update
            h = h * dA + rearrange(xt, 'b d -> b d 1') * dB
            
            # Output
            yt = torch.einsum('bdn,bn->bd', h, Ct)
            y.append(yt + xt * D)
            
        return torch.stack(y, dim=1)

class XFMambaClassifier(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, d_model=192, depth=8):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2) # [B, D, L]
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, d_model))
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                MambaBlock(d_model, d_state=16),
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).transpose(1, 2) # [B, L, D]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res # Residual connection
            
        x = self.norm(x)
        return self.head(x[:, 0])


# In[ ]:


# --- Dataset Loading ---
# --- Dataset Loading ---
# Robust path handling
if os.path.exists("./balanced_augmented_dataset"):
    dataset_root = os.path.abspath("./balanced_augmented_dataset")
else:
    # Use absolute path as fallback
    dataset_root = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\balanced_augmented_dataset"

print(f"Using dataset path: {dataset_root}")
train_dir = os.path.join(dataset_root, 'train')
val_dir = os.path.join(dataset_root, 'val')

IMG_SIZE = 224
BATCH_SIZE = 16 # Reduced batch size slightly for safer memory usage

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}")


# In[ ]:


# --- Training Loop ---
model = XFMambaClassifier(num_classes=num_classes).to(device)
print("XFMamba Model initialized.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

EPOCHS = 10
best_acc = 0.0
save_path = "../outputs/xfmamba_best_pure.pth"

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    train_acc = 100 * train_correct / train_total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Saved Best Model: {save_path}")

