#!/usr/bin/env python3
"""
Comprehensive benchmark of all models with detailed metrics.
Generates MODEL_BENCHMARK_RESULTS.md with accuracy, precision, recall, F1-score.
"""

import sys
sys.path.insert(0, '.')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime
import os

from src.utils import get_model, get_transforms, FractureDataset
from src.models import HyperColumnCBAMDenseNet169

# Configuration
NUM_CLASSES = 8
DATA_ROOT = 'data'
BATCH_SIZE = 16
IMG_SIZE = 224

CLASS_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Oblique_Displaced',
    'Oblique', 'Spiral', 'Transverse_Displaced', 'Transverse'
]

# All models to benchmark
MODELS = {
    'densenet169': {'file': 'best_densenet169.pth', 'type': 'standard'},
    'efficientnetv2': {'file': 'best_efficientnetv2.pth', 'type': 'standard'},
    'hypercolumn_cbam': {'file': 'best_hypercolumn_cbam_densenet169.pth', 'type': 'hypercolumn'},
    'maxvit': {'file': 'best_maxvit.pth', 'type': 'standard'},
    'mobilenetv2': {'file': 'best_mobilenetv2.pth', 'type': 'standard'},
    'swin': {'file': 'best_swin.pth', 'type': 'standard'},
}


def load_model(model_name, model_info, device):
    """Load a model from checkpoint."""
    model_path = f"models/{model_info['file']}"
    
    if not os.path.exists(model_path):
        return None, f"File not found: {model_path}"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model
        if model_info['type'] == 'hypercolumn':
            model = HyperColumnCBAMDenseNet169(num_classes=NUM_CLASSES, pretrained=False)
        else:
            model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        return model, None
    except Exception as e:
        return None, str(e)


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and labels."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred):
    """Compute all metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    
    metrics['per_class'] = {
        'precision': per_class_precision,
        'recall': per_class_recall,
        'f1': per_class_f1
    }
    
    return metrics


def generate_markdown(results, test_size):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"""# Model Benchmark Results

**Generated:** {timestamp}  
**Test Set Size:** {test_size} samples  
**Number of Classes:** {NUM_CLASSES}

## Summary

All models have been retrained with the corrected dataset labels.

### Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
"""
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['accuracy'], reverse=True)
    
    for model_name, data in sorted_results:
        m = data['metrics']
        md += f"| {model_name} | {m['accuracy']:.2f}% | {m['precision_macro']:.2f}% | {m['recall_macro']:.2f}% | {m['f1_macro']:.2f}% |\n"
    
    md += """
### Best Performing Models

"""
    
    # Top 3 models
    for i, (model_name, data) in enumerate(sorted_results[:3], 1):
        m = data['metrics']
        md += f"{i}. **{model_name}**: {m['accuracy']:.2f}% accuracy, {m['f1_macro']:.2f}% F1-score\n"
    
    md += """
## Detailed Metrics

### Weighted Metrics (accounts for class imbalance)

| Model | Precision (W) | Recall (W) | F1-Score (W) |
|-------|---------------|------------|--------------|
"""
    
    for model_name, data in sorted_results:
        m = data['metrics']
        md += f"| {model_name} | {m['precision_weighted']:.2f}% | {m['recall_weighted']:.2f}% | {m['f1_weighted']:.2f}% |\n"
    
    md += """
## Per-Class Performance

### Best Model Per-Class Breakdown
"""
    
    best_model_name, best_data = sorted_results[0]
    md += f"\n**{best_model_name}** (Top Performer)\n\n"
    md += "| Class | Precision | Recall | F1-Score |\n"
    md += "|-------|-----------|--------|----------|\n"
    
    per_class = best_data['metrics']['per_class']
    for i, class_name in enumerate(CLASS_NAMES):
        md += f"| {class_name} | {per_class['precision'][i]:.2f}% | {per_class['recall'][i]:.2f}% | {per_class['f1'][i]:.2f}% |\n"
    
    md += """
## Confusion Matrix Analysis

### Top Model Confusion Patterns
"""
    
    # Add confusion matrix for best model
    cm = confusion_matrix(best_data['labels'], best_data['preds'])
    md += f"\n**{best_model_name}** Confusion Matrix:\n\n"
    md += "```\n"
    md += "True\\Pred  " + "  ".join([f"{i:>3}" for i in range(NUM_CLASSES)]) + "\n"
    for i in range(NUM_CLASSES):
        md += f"   {i}       " + "  ".join([f"{cm[i,j]:>3}" for j in range(NUM_CLASSES)]) + f"  ({CLASS_NAMES[i]})\n"
    md += "```\n"
    
    md += """
## Model Architecture Summary

| Model | Architecture | Key Features |
|-------|--------------|--------------|
| densenet169 | DenseNet-169 | Dense connections, feature reuse |
| efficientnetv2 | EfficientNet-V2 | Compound scaling, fused MBConv |
| hypercolumn_cbam | DenseNet-169 + HyperColumn + CBAM | Multi-scale features, attention |
| maxvit | MaxViT | Multi-axis attention, hybrid CNN-Transformer |
| mobilenetv2 | MobileNet-V2 | Inverted residuals, lightweight |
| swin | Swin Transformer | Shifted windows, hierarchical |

## Training Configuration

- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Weight Decay:** 0.01
- **Scheduler:** CosineAnnealingLR
- **Batch Size:** 16
- **Image Size:** 224x224

## Notes

- All models were retrained after fixing label mismatches in the test set
- The label corrections affected classes: Oblique ↔ Oblique_Displaced, Transverse ↔ Transverse_Displaced
- Test set contains original (non-augmented) images only
- Training set contains augmented images for data balancing
"""
    
    return md


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
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv(f'{DATA_ROOT}/balanced_augmented_dataset/test.csv').to_dict('records')
    test_transform = get_transforms('test', IMG_SIZE)
    test_dataset = FractureDataset(test_df, DATA_ROOT, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    results = {}
    
    # Evaluate each model
    for model_name, model_info in MODELS.items():
        print(f"\nEvaluating {model_name}...")
        
        model, error = load_model(model_name, model_info, device)
        
        if error:
            print(f"  ❌ Error: {error}")
            continue
        
        preds, labels = evaluate_model(model, test_loader, device)
        metrics = compute_metrics(labels, preds)
        
        results[model_name] = {
            'metrics': metrics,
            'preds': preds,
            'labels': labels
        }
        
        print(f"  ✓ Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1_macro']:.2f}%")
    
    # Generate markdown
    print("\n" + "="*60)
    print("GENERATING BENCHMARK REPORT")
    print("="*60)
    
    md_content = generate_markdown(results, len(test_dataset))
    
    with open('MODEL_BENCHMARK_RESULTS.md', 'w') as f:
        f.write(md_content)
    
    print("\n✅ Benchmark complete! Results saved to MODEL_BENCHMARK_RESULTS.md")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['accuracy'], reverse=True)
    for model_name, data in sorted_results:
        m = data['metrics']
        print(f"{model_name:<25} {m['accuracy']:>9.2f}% {m['precision_macro']:>9.2f}% {m['recall_macro']:>9.2f}% {m['f1_macro']:>9.2f}%")


if __name__ == '__main__':
    main()
