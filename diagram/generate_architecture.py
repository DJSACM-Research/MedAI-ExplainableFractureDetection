import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
c_input = '#B3E5FC'
c_backbone = '#FFCCBC'
c_head = '#C5E1A5'
c_post = '#E1BEE7'
c_output = '#FFF9C4'

def box(ax, x, y, w, h, text, color, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                      ec='black', fc=color, lw=1.5, zorder=2)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, weight='bold', zorder=3)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Title
ax.text(7, 9.5, 'Deep Learning Architecture', ha='center', fontsize=16, weight='bold')
ax.text(7, 9.1, 'Vision Transformers for Fracture Classification', ha='center', fontsize=11, style='italic')

# Input
box(ax, 1, 7.5, 1.5, 1, 'X-Ray\nImage\n224×224', c_input)
arrow(ax, 2.5, 8, 3.5, 8)

# Preprocessing
box(ax, 3.5, 7.5, 1.8, 1, 'Normalize\nResize\nAugment', c_input, fs=8)
arrow(ax, 5.3, 8, 6.3, 8)

# Backbone Options
ax.text(9, 8.5, 'Backbone Architecture', fontsize=12, weight='bold', ha='center')
box(ax, 6.5, 7.2, 2, 0.6, 'Swin Transformer', c_backbone, fs=8)
box(ax, 6.5, 6.4, 2, 0.6, 'ConvNeXt', c_backbone, fs=8)
box(ax, 6.5, 5.6, 2, 0.6, 'DenseNet-169', c_backbone, fs=8)

arrow(ax, 8.5, 6.8, 9.5, 6)

# Feature Extraction
box(ax, 9.5, 5.2, 2.5, 1.5, 'Feature\nExtraction\n(2048-d)', c_head)
arrow(ax, 10.75, 5.2, 10.75, 4.2)

# Classification Head
box(ax, 9.5, 3, 2.5, 1, 'FC Layer\n8 Classes', c_head)
arrow(ax, 10.75, 3, 10.75, 2)

# Softmax
box(ax, 9.5, 1.2, 2.5, 0.6, 'Softmax', c_output)

# Grad-CAM Branch
ax.text(3.5, 4.5, 'Explainability Branch', fontsize=11, weight='bold', color='#6A1B9A')
arrow(ax, 7.5, 5.9, 5.5, 4.5)
box(ax, 3.5, 3.8, 2, 0.6, 'Grad-CAM', c_post, fs=8)
arrow(ax, 4.5, 3.8, 4.5, 3.2)
box(ax, 3.5, 2.5, 2, 0.6, 'Heatmap\nOverlay', c_post, fs=8)

# Outputs
ax.text(7, 0.8, 'Outputs', fontsize=12, weight='bold', ha='center')
box(ax, 9.5, 0.3, 2.5, 0.7, 'Predicted Class\n+ Confidence', c_output, fs=9)
box(ax, 3.5, 1.6, 2, 0.7, 'Visual\nExplanation', c_output, fs=8)

# Loss function annotation
box(ax, 0.5, 1, 2.5, 1.2, 'Training Loss:\n• Cross-Entropy\n• Focal Loss\n• Class Weights', '#FFF3E0', fs=7)

# Metrics
box(ax, 0.5, 2.5, 2.5, 1, 'Metrics:\n• Macro F1-Score\n• Per-Class Recall\n• Confusion Matrix', '#E0F2F1', fs=7)

# Model specs
ax.text(12.5, 7, 'Model Specs:', fontsize=9, weight='bold', ha='left')
specs = [
    'Parameters: 28M-50M',
    'Input: 224×224 RGB',
    'Classes: 8',
    'Batch Size: 4-8',
    'Optimizer: AdamW',
    'LR: 1e-4',
    'Device: MPS/CUDA'
]
for i, spec in enumerate(specs):
    ax.text(12.5, 6.5-i*0.35, f'• {spec}', fontsize=7, ha='left')

plt.tight_layout()
plt.savefig('diagram/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Architecture diagram saved to: diagram/architecture_diagram.png")
plt.close()
