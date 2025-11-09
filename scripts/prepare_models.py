#!/usr/bin/env python3
"""Verify and prepare models for deployment."""

import os
import sys
from pathlib import Path

def check_models():
    """Check if all required models exist."""
    output_dir = Path("./outputs")
    
    if not output_dir.exists():
        print("❌ ERROR: ./outputs directory does not exist")
        return False
    
    required_models = [
        "best_swin.pth",
        "best_mobilenetv2.pth",
        "best_densenet169.pth",
        "best_efficientnetv2.pth",
        "best_maxvit.pth"
    ]
    
    found_models = []
    missing_models = []
    
    for model in required_models:
        model_path = output_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model}: {size_mb:.1f} MB")
            found_models.append(model)
        else:
            print(f"❌ {model}: NOT FOUND")
            missing_models.append(model)
    
    print(f"\n📊 Summary: {len(found_models)}/{len(required_models)} models found")
    
    if missing_models:
        print(f"\n⚠️  Missing models:")
        for model in missing_models:
            print(f"   - {model}")
        return False
    
    return True

if __name__ == "__main__":
    if not check_models():
        sys.exit(1)
    print("\n✅ All models are ready for deployment!")
