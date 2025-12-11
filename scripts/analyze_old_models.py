#!/usr/bin/env python3
"""Analyze why hypercolumn and swin models have low accuracy."""

import torch
import sys
sys.path.insert(0, '.')

from src.utils import get_model

print("="*60)
print("HYPERCOLUMN MODEL ANALYSIS")
print("="*60)

# Load the checkpoint
checkpoint = torch.load('models/best_hypercolumn_cbam_densenet169.pth', map_location='cpu', weights_only=False)
ckpt_keys = set(checkpoint.keys())

# Create the model
model = get_model('hypercolumn_densenet169', num_classes=8, pretrained=False)
model_keys = set(model.state_dict().keys())

# Compare keys
missing_in_ckpt = model_keys - ckpt_keys
extra_in_ckpt = ckpt_keys - model_keys

print(f"\nCheckpoint keys: {len(ckpt_keys)}")
print(f"Model keys: {len(model_keys)}")
print(f"\nMissing in checkpoint: {len(missing_in_ckpt)}")
for k in list(missing_in_ckpt)[:10]:
    print(f"  {k}")

print(f"\nExtra in checkpoint: {len(extra_in_ckpt)}")
for k in list(extra_in_ckpt)[:10]:
    print(f"  {k}")

# Try to load
print("\n" + "="*60)
print("Attempting to load weights...")
try:
    model.load_state_dict(checkpoint, strict=True)
    print("SUCCESS: Weights loaded with strict=True")
except Exception as e:
    print(f"FAILED strict=True: {str(e)[:100]}")
    
# Now check the Swin model
print("\n" + "="*60)
print("SWIN MODEL ANALYSIS")
print("="*60)

swin_ckpt = torch.load('models/best_swin.pth', map_location='cpu', weights_only=False)
if 'model_state_dict' in swin_ckpt:
    swin_state = swin_ckpt['model_state_dict']
else:
    swin_state = swin_ckpt

swin_ckpt_keys = set(swin_state.keys())
swin_model = get_model('swin', num_classes=8, pretrained=False)
swin_model_keys = set(swin_model.state_dict().keys())

print(f"\nCheckpoint keys: {len(swin_ckpt_keys)}")
print(f"Model keys: {len(swin_model_keys)}")

# These models were trained BEFORE the test labels were fixed
# So they were trained with the WRONG test labels being used for evaluation
# But they should still work correctly now
print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("""
These models (hypercolumn_cbam and swin) were trained BEFORE 
we fixed the test labels. They are the ORIGINAL models that 
existed in the repository.

The question is: were they trained with a different label mapping?

Let's check if the checkpoint has any metadata about training...
""")

# Check for any training metadata
print("Swin checkpoint keys:")
if isinstance(swin_ckpt, dict):
    for k in swin_ckpt.keys():
        if k != 'model_state_dict':
            print(f"  {k}: {swin_ckpt[k]}")
