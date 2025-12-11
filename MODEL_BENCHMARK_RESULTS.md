# Model Benchmark Results

**Generated:** 2025-12-11 19:07:53  
**Test Set Size:** 112 samples  
**Number of Classes:** 8

## Summary

All models have been retrained with the corrected dataset labels.

### Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| maxvit | 91.07% | 91.75% | 92.65% | 91.99% |
| densenet169 | 90.18% | 90.10% | 91.91% | 90.74% |
| hypercolumn_cbam | 87.50% | 87.65% | 89.05% | 88.14% |
| efficientnetv2 | 86.61% | 87.56% | 88.32% | 87.55% |
| mobilenetv2 | 85.71% | 85.64% | 87.36% | 86.02% |
| swin | 85.71% | 86.08% | 87.58% | 86.29% |

### Best Performing Models

1. **maxvit**: 91.07% accuracy, 91.99% F1-score
2. **densenet169**: 90.18% accuracy, 90.74% F1-score
3. **hypercolumn_cbam**: 87.50% accuracy, 88.14% F1-score

## Detailed Metrics

### Weighted Metrics (accounts for class imbalance)

| Model | Precision (W) | Recall (W) | F1-Score (W) |
|-------|---------------|------------|--------------|
| maxvit | 91.03% | 91.07% | 90.83% |
| densenet169 | 90.08% | 90.18% | 89.88% |
| hypercolumn_cbam | 87.09% | 87.50% | 87.09% |
| efficientnetv2 | 86.48% | 86.61% | 86.15% |
| mobilenetv2 | 85.26% | 85.71% | 84.96% |
| swin | 86.28% | 85.71% | 85.45% |

## Per-Class Performance

### Best Model Per-Class Breakdown

**maxvit** (Top Performer)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Comminuted | 94.44% | 100.00% | 97.14% |
| Greenstick | 86.67% | 100.00% | 92.86% |
| Healthy | 90.91% | 100.00% | 95.24% |
| Oblique_Displaced | 92.86% | 76.47% | 83.87% |
| Oblique | 100.00% | 100.00% | 100.00% |
| Spiral | 100.00% | 100.00% | 100.00% |
| Transverse_Displaced | 75.00% | 70.59% | 72.73% |
| Transverse | 94.12% | 94.12% | 94.12% |

## Confusion Matrix Analysis

### Top Model Confusion Patterns

**maxvit** Confusion Matrix:

```
True\Pred    0    1    2    3    4    5    6    7
   0        17    0    0    0    0    0    0    0  (Comminuted)
   1         0   13    0    0    0    0    0    0  (Greenstick)
   2         0    0   10    0    0    0    0    0  (Healthy)
   3         0    0    0   13    0    0    4    0  (Oblique_Displaced)
   4         0    0    0    0    9    0    0    0  (Oblique)
   5         0    0    0    0    0   12    0    0  (Spiral)
   6         1    2    1    0    0    0   12    1  (Transverse_Displaced)
   7         0    0    0    1    0    0    0   16  (Transverse)
```

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
