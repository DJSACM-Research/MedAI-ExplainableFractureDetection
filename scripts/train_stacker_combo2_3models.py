#!/usr/bin/env python3
"""
Train Stacking Meta-Model for Combo 2 with 3 Models
Models: maxvit, hypercolumn_cbam_densenet169_focal, rad_dino
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from medai.modules.ensemble_module import EnsembleModule

# Configuration
DATASET_ROOT = Path("balanced_augmented_dataset")
TRAIN_DIR = DATASET_ROOT / "train"
OUTPUT_DIR = Path("outputs")

# Model names for Combo 2
MODEL_NAMES = ["maxvit", "hypercolumn_cbam_densenet169_focal", "rad_dino"]

# Class names (must match the actual classes)
CLASS_NAMES = [
    "Comminuted",
    "Greenstick",
    "Healthy",
    "Oblique",
    "Oblique_Displaced",
    "Spiral",
    "Transverse",
    "Transverse_Displaced",
]

def get_training_data():
    """Load training images and labels"""
    images = []
    labels = []
    label_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    
    for class_name in CLASS_NAMES:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ Class directory not found: {class_dir}")
            continue
            
        # Get all image files
        img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        for img_file in img_files:
            images.append(str(img_file))
            labels.append(label_to_idx[class_name])
    
    return np.array(images), np.array(labels)

def get_predictions_from_models(images, labels):
    """Get predictions from all 3 models on training data"""
    print("=" * 70)
    print("TRAINING STACKING META-MODEL FOR COMBO 2 (3 MODELS)")
    print("=" * 70)
    print(f"Models: {', '.join(MODEL_NAMES)}")
    print(f"Training samples: {len(images)}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print()
    
    # Initialize ensemble with 3 models
    print("Initializing Ensemble with 3 models...")
    ensemble = EnsembleModule(
        model_names=MODEL_NAMES,
        class_names=CLASS_NAMES,
        checkpoints_dir=OUTPUT_DIR / "cross_validation",
        device="cpu"
    )
    print("✅ Ensemble initialized")
    print()
    
    # Get predictions from each model
    all_predictions = []
    
    for model_name in MODEL_NAMES:
        print(f"\nGetting predictions from {model_name}...")
        model_predictions = []
        
        for i, img_path in enumerate(tqdm(images)):
            try:
                pred = ensemble._get_model_prediction(img_path, model_name)
                model_predictions.append(pred)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                model_predictions.append(np.zeros(len(CLASS_NAMES)) / len(CLASS_NAMES))
        
        all_predictions.append(np.array(model_predictions))
    
    # Stack predictions: shape (num_images, num_models * num_classes)
    stacked_features = np.hstack(all_predictions)
    print(f"\n✅ Stacked features shape: {stacked_features.shape}")
    
    return stacked_features, labels

def train_stacker(X, y):
    """Train Logistic Regression meta-model"""
    print("\n" + "=" * 70)
    print("TRAINING META-MODEL")
    print("=" * 70)
    
    # Create pipeline with StandardScaler and LogisticRegression
    stacker = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            verbose=1
        ))
    ])
    
    print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
    stacker.fit(X, y)
    
    # Evaluate on training data
    train_acc = stacker.score(X, y)
    print(f"✅ Training accuracy: {train_acc:.4f}")
    
    return stacker

def save_stacker(stacker, output_path):
    """Save trained stacker"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(stacker, output_path)
    print(f"✅ Stacker saved to: {output_path}")

def main():
    """Main training pipeline"""
    try:
        # Load training data
        print("Loading training data...")
        images, labels = get_training_data()
        print(f"✅ Loaded {len(images)} training samples")
        
        # Get predictions from models
        X, y = get_predictions_from_models(images, labels)
        
        # Train meta-model
        stacker = train_stacker(X, y)
        
        # Save stacker
        output_path = OUTPUT_DIR / "stacker_combo2_3models.joblib"
        save_stacker(stacker, output_path)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Stacker for Combo 2 (3 models) saved successfully!")
        print(f"Use this stacker for optimal Combo 2 performance")
        print()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
