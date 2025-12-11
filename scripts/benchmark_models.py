#!/usr/bin/env python3
"""
Benchmark All Models Script
Calculates accuracy, precision, recall, F1 for each model in the models/ directory.
Generates a comprehensive comparison report.
"""

import os
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_device, get_model, get_transforms


# Configuration
MODELS_DIR = "models"
DATA_ROOT = "data"
TEST_CSV = "data/balanced_augmented_dataset/test.csv"
CLASS_NAMES = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
               "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
OUTPUT_DIR = "outputs/benchmark_results"
REPORT_FILE = "MODEL_BENCHMARK_RESULTS.md"

# Map checkpoint files to model architecture names
MODEL_MAPPING = {
    "best_densenet169.pth": "densenet",
    "best_efficientnetv2.pth": "efficientnet",
    "best_hypercolumn_cbam_densenet169.pth": "hypercolumn_densenet169",
    "best_hypercolumn_cbam_densenet169_focal.pth": "hypercolumn_densenet169",
    "best_hypercolumn_cbam_densenet169 copy.pth": "hypercolumn_densenet169",
    "best_hypercolumn_densenet169.pth": "hypercolumn_densenet169",
    "best_hypercolumn_densenet169_old.pth": "hypercolumn_densenet169",
    "best_maxvit.pth": "maxvit",
    "best_mobilenetv2.pth": "mobilenet",
    "best_swin.pth": "swin",
}


def detect_architecture_from_checkpoint(checkpoint_path: str) -> str:
    """Auto-detect model architecture from checkpoint state dict keys."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    keys = list(state_dict.keys())
    first_keys = ' '.join(keys[:10])
    
    # Detect based on key patterns
    if any('patch_embed' in k and 'layers.' in k for k in keys):
        # Swin Transformer pattern
        return 'swin'
    elif any('densenet' in k.lower() or 'denseblock' in k for k in keys):
        return 'densenet'
    elif any('stages.' in k and 'attn_block' in k for k in keys):
        # MaxViT pattern
        return 'maxvit'
    elif any('blocks.' in k and 'se.conv' in k for k in keys):
        # EfficientNet pattern
        return 'efficientnet'
    elif any('blocks.' in k and 'conv_pw' in k for k in keys):
        # MobileNet pattern
        return 'mobilenet'
    elif any('cbam' in k.lower() or 'hypercolumn' in k.lower() for k in keys):
        return 'hypercolumn_densenet169'
    elif any('densenet' in k.lower() for k in keys):
        return 'hypercolumn_densenet169'
    elif 'features.conv0.weight' in keys or 'features.denseblock1' in first_keys:
        return 'densenet'
    
    return None


def load_test_data(csv_path: str):
    """Load test data from CSV file."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def evaluate_model(model, test_data, transform, device, data_root: str):
    """Evaluate a single model on test data."""
    model.eval()
    predictions = []
    true_labels = []
    all_probs = []
    
    with torch.no_grad():
        for row in test_data:
            img_path = row['image_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(data_root, img_path)
            
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                
                output = model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                
                predictions.append(pred)
                true_labels.append(int(row['label']))
                all_probs.append(probs)
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")
                continue
    
    return np.array(predictions), np.array(true_labels), np.array(all_probs)


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate all metrics for a model."""
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        average=None, 
        labels=list(range(len(class_names))),
        zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]) if support is not None else int(cm[i].sum())
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm
    }


def save_confusion_matrix(cm, class_names, output_path):
    """Save confusion matrix as an image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black', fontsize=8)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_report(all_results: dict, output_file: str):
    """Generate a comprehensive markdown report."""
    with open(output_file, 'w') as f:
        f.write("# Model Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Test Dataset:** {TEST_CSV}\n\n")
        f.write(f"**Number of Classes:** {len(CLASS_NAMES)}\n\n")
        
        # Summary Table
        f.write("## Summary Comparison\n\n")
        f.write("| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Weighted F1 |\n")
        f.write("|-------|----------|----------|-----------------|--------------|-------------|\n")
        
        # Sort by macro F1
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
        
        for model_name, metrics in sorted_results:
            f.write(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | ")
            f.write(f"{metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | ")
            f.write(f"{metrics['weighted_f1']:.4f} |\n")
        
        f.write("\n")
        
        # Best model highlight
        if sorted_results:
            best_model, best_metrics = sorted_results[0]
            f.write(f"### 🏆 Best Model: **{best_model}**\n")
            f.write(f"- Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)\n")
            f.write(f"- Macro F1: {best_metrics['macro_f1']:.4f}\n\n")
        
        # Per-Model Details
        f.write("---\n\n## Detailed Results by Model\n\n")
        
        for model_name, metrics in sorted_results:
            f.write(f"### {model_name}\n\n")
            f.write(f"**Overall Metrics:**\n")
            f.write(f"- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"- Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"- Macro Recall: {metrics['macro_recall']:.4f}\n")
            f.write(f"- Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"- Weighted F1: {metrics['weighted_f1']:.4f}\n\n")
            
            f.write("**Per-Class Metrics:**\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"| {class_name} | {class_metrics['precision']:.4f} | ")
                f.write(f"{class_metrics['recall']:.4f} | {class_metrics['f1']:.4f} | ")
                f.write(f"{class_metrics['support']} |\n")
            
            f.write("\n")
            
            # Link to confusion matrix
            safe_name = model_name.replace(" ", "_").replace("/", "_")
            f.write(f"![Confusion Matrix]({OUTPUT_DIR}/{safe_name}_confusion_matrix.png)\n\n")
            f.write("---\n\n")
    
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument('--models-dir', default=MODELS_DIR, help='Directory containing model checkpoints')
    parser.add_argument('--test-csv', default=TEST_CSV, help='Path to test CSV file')
    parser.add_argument('--data-root', default=DATA_ROOT, help='Root directory for image paths')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--report', default=REPORT_FILE, help='Output report filename')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_csv}...")
    test_data = load_test_data(args.test_csv)
    print(f"Loaded {len(test_data)} test samples")
    
    # Get transforms
    transform = get_transforms('val', 224)
    
    # Find all model checkpoints
    model_files = [f for f in os.listdir(args.models_dir) if f.endswith('.pth')]
    print(f"\nFound {len(model_files)} model checkpoints in {args.models_dir}/")
    
    all_results = {}
    failed_models = []
    
    for checkpoint_file in sorted(model_files):
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_file}")
        print('='*60)
        
        # Get model architecture name - try auto-detection first, fallback to mapping
        model_arch = None
        try:
            model_arch = detect_architecture_from_checkpoint(checkpoint_path)
            if model_arch:
                print(f"  Auto-detected architecture: {model_arch}")
        except Exception as e:
            print(f"  Could not auto-detect architecture: {e}")
        
        # Fallback to mapping
        if model_arch is None:
            model_arch = MODEL_MAPPING.get(checkpoint_file)
        
        if model_arch is None:
            print(f"  ⚠️  Unknown model architecture for {checkpoint_file}, skipping...")
            failed_models.append((checkpoint_file, "Unknown architecture"))
            continue
        
        checkpoint_path = os.path.join(args.models_dir, checkpoint_file)
        
        try:
            # Load model
            print(f"  Loading model architecture: {model_arch}")
            model = get_model(model_arch, num_classes=len(CLASS_NAMES), pretrained=False)
            
            # Load checkpoint
            print(f"  Loading checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            
            model.to(device)
            model.eval()
            
            # Evaluate
            print(f"  Running inference on {len(test_data)} samples...")
            predictions, true_labels, probs = evaluate_model(
                model, test_data, transform, device, args.data_root
            )
            
            # Calculate metrics
            print(f"  Calculating metrics...")
            metrics = calculate_metrics(predictions, true_labels, CLASS_NAMES)
            
            # Display summary
            print(f"\n  Results for {checkpoint_file}:")
            print(f"    Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"    Macro Precision: {metrics['macro_precision']:.4f}")
            print(f"    Macro Recall:    {metrics['macro_recall']:.4f}")
            print(f"    Macro F1:        {metrics['macro_f1']:.4f}")
            print(f"    Weighted F1:     {metrics['weighted_f1']:.4f}")
            
            # Save confusion matrix
            safe_name = checkpoint_file.replace('.pth', '').replace(" ", "_")
            cm_path = os.path.join(args.output_dir, f"{safe_name}_confusion_matrix.png")
            save_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, cm_path)
            print(f"  Confusion matrix saved to: {cm_path}")
            
            # Store results
            display_name = checkpoint_file.replace('.pth', '').replace('best_', '').replace('_', ' ').title()
            all_results[display_name] = metrics
            
        except Exception as e:
            print(f"  ❌ Error evaluating {checkpoint_file}: {e}")
            failed_models.append((checkpoint_file, str(e)))
            continue
    
    # Generate report
    print(f"\n{'='*60}")
    print("Generating benchmark report...")
    print('='*60)
    generate_report(all_results, args.report)
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print('='*60)
    print(f"✅ Successfully evaluated: {len(all_results)} models")
    if failed_models:
        print(f"❌ Failed: {len(failed_models)} models")
        for name, error in failed_models:
            print(f"   - {name}: {error}")
    
    print(f"\n📊 Results saved to: {args.report}")
    print(f"📁 Confusion matrices saved to: {args.output_dir}/")
    
    # Print ranking
    if all_results:
        print(f"\n🏆 Model Ranking (by Macro F1):")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
        for i, (name, metrics) in enumerate(sorted_results, 1):
            print(f"   {i}. {name}: {metrics['macro_f1']:.4f} (Acc: {metrics['accuracy']*100:.2f}%)")


if __name__ == '__main__':
    main()
