#!/usr/bin/env python3
"""Inspect model checkpoint files to determine their actual architectures."""

import torch
import os

def inspect_models():
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    results = []
    
    for model_file in sorted(model_files):
        path = os.path.join(models_dir, model_file)
        print(f'\n{"="*70}')
        print(f'FILE: {model_file}')
        print("="*70)
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Get state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Checkpoint type: dict with 'model_state_dict'")
            if 'epoch' in checkpoint:
                print(f"Trained epochs: {checkpoint.get('epoch', 'N/A')}")
            if 'best_acc' in checkpoint:
                print(f"Best accuracy: {checkpoint.get('best_acc', 'N/A'):.4f}")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            print(f"Checkpoint type: raw state_dict")
        else:
            print(f"Checkpoint type: {type(checkpoint)}")
            continue
        
        keys = list(state_dict.keys())
        print(f"Total parameters: {len(keys)}")
        
        print(f"\nFirst 10 layer keys:")
        for k in keys[:10]:
            shape = tuple(state_dict[k].shape) if hasattr(state_dict[k], 'shape') else 'N/A'
            print(f"  {k}: {shape}")
        
        print(f"\nLast 5 layer keys:")
        for k in keys[-5:]:
            shape = tuple(state_dict[k].shape) if hasattr(state_dict[k], 'shape') else 'N/A'
            print(f"  {k}: {shape}")
        
        # Detect architecture based on key patterns
        all_keys = ' '.join(keys)
        
        if 'backbone.features' in all_keys and 'cbam' in all_keys.lower():
            arch = "HyperColumn-CBAM-DenseNet169"
        elif 'backbone.features' in all_keys and 'fusion' in all_keys:
            arch = "HyperColumn-DenseNet169 (without CBAM)"
        elif 'patch_embed' in all_keys and 'layers.0.blocks' in all_keys:
            arch = "Swin Transformer"
        elif 'features.denseblock' in all_keys:
            arch = "DenseNet (standard timm/torchvision)"
        elif 'blocks' in all_keys and 'stem.conv' in all_keys:
            if 'stages' in all_keys:
                arch = "MaxViT"
            else:
                arch = "EfficientNet"
        elif 'features.0.0.weight' in all_keys:
            arch = "MobileNetV2"
        elif 'stages' in all_keys and 'stem.0' in all_keys:
            arch = "ConvNeXt"
        else:
            arch = "Unknown"
        
        # Get number of output classes from final layer
        num_classes = None
        for k in reversed(keys):
            if 'classifier' in k or 'head' in k or 'fc' in k:
                if 'weight' in k:
                    num_classes = state_dict[k].shape[0]
                    break
        
        print(f"\nNumber of output classes: {num_classes}")
        print(f"\n>>> DETECTED ARCHITECTURE: {arch}")
        
        results.append({
            'file': model_file,
            'arch': arch,
            'num_params': len(keys),
            'num_classes': num_classes
        })
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'File':<45} {'Architecture':<25}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<45} {r['arch']:<25}")

if __name__ == '__main__':
    inspect_models()
