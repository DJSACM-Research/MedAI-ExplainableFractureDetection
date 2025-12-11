#!/usr/bin/env python3
"""
Verify why old models (hypercolumn_cbam, swin) have ~48% accuracy.
Check if they were trained with incorrect label mappings.
"""
import torch
import sys
sys.path.insert(0, '.')

# Just print checkpoint keys, not full tensors
import torch
swin = torch.load('models/best_swin.pth', map_location='cpu', weights_only=False)
print("SWIN CHECKPOINT KEYS:", list(swin.keys()))
for k, v in swin.items():
    if k != 'model_state_dict':
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <{type(v).__name__}>")

print()
hc = torch.load('models/best_hypercolumn_cbam_densenet169.pth', map_location='cpu', weights_only=False)
print("HYPERCOLUMN CHECKPOINT KEYS:", list(hc.keys()))
for k, v in hc.items():
    if k != 'model_state_dict':
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <{type(v).__name__}>")
