# Quick Reference - Restructured Codebase

## Import Cheat Sheet

### Utilities

```python
from src.utils import get_device, get_model, get_transforms, FractureDataset
from src.utils import require_mps  # For MPS-only scripts
```

### Agents

```python
from src.agents import (
    DiagnosticAgent,
    EducationalAgent,
    KnowledgeAgent,
    ModelEnsembleAgent,
    generate_random_heatmap,
    calculate_heatmap_centroid
)
```

## Common Patterns

### Device Detection

```python
from src.utils import get_device
device = get_device()  # Auto-detects: cuda > mps > cpu
```

### Model Loading

```python
from src.utils import get_model

# Load pretrained model
model = get_model('swin', num_classes=8, pretrained=True)

# Load model for inference (no pretrained weights)
model = get_model('swin', num_classes=8, pretrained=False)

# Supported models: 'swin', 'convnext', 'densenet'
```

### Data Transforms

```python
from src.utils import get_transforms

# Training transforms (with augmentation)
train_tf = get_transforms('train', img_size=224)

# Validation/test transforms (no augmentation)
val_tf = get_transforms('val', img_size=224)
```

### Dataset Creation

```python
from src.utils import FractureDataset, get_transforms
import pandas as pd

df = pd.read_csv('data/train.csv')
transforms = get_transforms('train', 224)

dataset = FractureDataset(
    df=df,
    img_root='data',
    transform=transforms,
    use_bbox=False  # Set True to crop using bbox columns
)

# Dataset returns: (image_tensor, label, image_path)
img, label, path = dataset[0]
```

## Running Scripts

### Training

```bash
# Pipeline 1 (MPS only)
python3 src/training/pipeline.py \
    --train-csv data/balanced_augmented_dataset/train.csv \
    --val-csv data/balanced_augmented_dataset/val.csv \
    --model swin \
    --epochs 20 \
    --batch-size 6

# Pipeline 2 (CUDA/MPS/CPU)
python3 src/training/pipeline_2.py \
    --train-csv data/balanced_augmented_dataset/train.csv \
    --val-csv data/balanced_augmented_dataset/val.csv \
    --model swin
```

### Analysis

```bash
# Basic analysis
python3 src/analysis/analyze.py \
    --checkpoint outputs/swin_mps/best.pth \
    --test-csv data/balanced_augmented_dataset/test.csv \
    --model swin \
    --class-names "Comminuted,Greenstick,Healthy,..."

# Analysis with Grad-CAM
python3 src/analysis/analyze_2.py \
    --checkpoint outputs/swin_mps/best.pth \
    --test-csv data/balanced_augmented_dataset/test.csv \
    --model swin

# Visualize Grad-CAM
python3 src/analysis/visualize_gradcam.py \
    --checkpoint outputs/swin_mps/best.pth \
    --misclassified outputs/analysis/misclassified.csv \
    --model swin
```

### Agents

```bash
# Diagnostic agent
python3 src/agents/diagnostic_agent.py \
    --image-path path/to/xray.jpg \
    --checkpoint outputs/swin_mps/best.pth \
    --model swin

# Cross-validation ensemble
python3 src/agents/cross_validation_agent.py \
    --image-path path/to/xray.jpg \
    --checkpoints-dir outputs/ensemble/ \
    --model-names swin,convnext,densenet
```

## Supported Models

| Model Name | Architecture           | Input Size | Parameters |
| ---------- | ---------------------- | ---------- | ---------- |
| `swin`     | Swin Transformer Small | 224x224    | ~50M       |
| `convnext` | ConvNeXt Tiny          | 224x224    | ~28M       |
| `densenet` | DenseNet-169           | 224x224    | ~14M       |

## File Locations

| Purpose           | Location        |
| ----------------- | --------------- |
| Shared utilities  | `src/utils/`    |
| AI agents         | `src/agents/`   |
| Training scripts  | `src/training/` |
| Analysis scripts  | `src/analysis/` |
| Datasets          | `data/`         |
| Model checkpoints | `outputs/`      |
| Notebooks         | `notebooks/`    |
| Documentation     | `docs/`         |

## Validation

```bash
# Check structure and syntax
python3 src/validate_structure.py

# Test functionality (requires dependencies)
python3 src/test_migration.py
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'src'`:

```python
# Add this at the top of your script
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Device Issues

```python
from src.utils import get_device
print(f"Using device: {get_device()}")  # Check what device is selected
```

### Model Loading Issues

```python
# Use strict=False for partial weight loading
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

---

**Quick Start**: `python3 src/validate_structure.py` to verify everything is set up correctly.
