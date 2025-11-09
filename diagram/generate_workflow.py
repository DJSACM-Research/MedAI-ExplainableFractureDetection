import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_data = '#E8F4F8'
color_training = '#FFF4E6'
color_inference = '#E8F5E9'
color_agents = '#F3E5F5'
color_output = '#FCE4EC'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10, style='round'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"{style},pad=0.1",
                         edgecolor='black', facecolor=color,
                         linewidth=2, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize,
           weight='bold', zorder=3)
    return box

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color='black',
                           linewidth=2, zorder=1,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=8,
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# Title
ax.text(8, 11.5, 'MedAI: Explainable Fracture Detection System',
       ha='center', fontsize=18, weight='bold')
ax.text(8, 11, 'Multi-Agent Architecture for Medical Imaging Analysis',
       ha='center', fontsize=12, style='italic')

# ============= PHASE 1: DATA & TRAINING =============
ax.text(2, 10, 'Phase 1: Data & Training', fontsize=14, weight='bold', color='#D84315')

# Data Input
create_box(ax, 0.5, 8.5, 1.5, 0.8, 'Raw\nDataset', color_data)
create_arrow(ax, 2.25, 8.9, 3.25, 8.9)

# EDA
create_box(ax, 3.5, 8.5, 1.5, 0.8, 'EDA\nNotebook', color_data)
create_arrow(ax, 5.25, 8.9, 6.25, 8.9)

# Preprocessing
create_box(ax, 6.5, 8.5, 1.8, 0.8, 'Augmentation\n& Balancing', color_data)

# Training Pipeline
create_arrow(ax, 7.4, 8.5, 7.4, 7.5)
create_box(ax, 5.5, 6.5, 3.8, 0.8, 'Training Pipeline\n(Swin/ConvNext/DenseNet)', color_training)

# Checkpoints
create_arrow(ax, 7.4, 6.5, 7.4, 5.5)
create_box(ax, 6, 4.5, 2.8, 0.8, 'Best Checkpoint\n(best.pth)', color_training)

# ============= PHASE 2: INFERENCE & AGENTS =============
ax.text(11, 10, 'Phase 2: Inference & Multi-Agent System', fontsize=14, weight='bold', color='#1565C0')

# Input Image
create_box(ax, 10.5, 8.5, 1.5, 0.8, 'X-Ray\nImage', color_inference)
create_arrow(ax, 12, 8.5, 12, 7.5)

# Agent 1: Diagnostic
create_box(ax, 10.5, 6.5, 3, 0.8, '① Diagnostic Agent\n(Classification)', color_agents)
create_arrow(ax, 12, 6.5, 12, 5.5)

# Agent 2: Explainability
create_box(ax, 10.5, 4.5, 3, 0.8, '② Explainability Agent\n(Grad-CAM)', color_agents)
create_arrow(ax, 12, 4.5, 12, 3.5)

# Agent 3: Knowledge
create_box(ax, 10.5, 2.5, 3, 0.8, '③ Knowledge Agent\n(Medical KB)', color_agents)

# Agent 4: Educational
create_arrow(ax, 12, 2.5, 12, 1.5)
create_box(ax, 10.5, 0.5, 3, 0.8, '④ Educational Agent\n(Patient Translation)', color_agents)

# ============= PHASE 3: OUTPUTS =============
ax.text(2, 5.5, 'Phase 3: Outputs & Applications', fontsize=14, weight='bold', color='#2E7D32')

# Analysis Output
create_box(ax, 0.5, 3.5, 2, 0.8, 'Analysis\nReports', color_output)

# Grad-CAM Visualizations
create_box(ax, 0.5, 2.3, 2, 0.8, 'Grad-CAM\nOverlays', color_output)

# Patient Chat
create_box(ax, 0.5, 1.1, 2, 0.8, 'Patient Chat\nApp (Llama 3)', color_output)

# Cross connections
create_arrow(ax, 7.4, 4.5, 10.5, 6.9, label='Model Weights')
create_arrow(ax, 9.3, 5.5, 8, 4.1, label='Analysis', style='->')
create_arrow(ax, 13.5, 0.9, 2.5, 1.5, label='RAG Context', style='->')

# ============= ENSEMBLE BOX =============
create_box(ax, 10, 0, 4.5, 0.3, 'Ensemble Agent (Multi-Model Voting)', '#FFE0B2', fontsize=9)

# ============= LEGEND =============
legend_y = 0.2
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Data Processing'),
    mpatches.Patch(facecolor=color_training, edgecolor='black', label='Training'),
    mpatches.Patch(facecolor=color_inference, edgecolor='black', label='Inference Input'),
    mpatches.Patch(facecolor=color_agents, edgecolor='black', label='AI Agents'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='User-Facing Outputs')
]
ax.legend(handles=legend_elements, loc='lower left', ncol=5, fontsize=9, frameon=True)

# Save
plt.tight_layout()
plt.savefig('diagram/workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Workflow diagram saved to: diagram/workflow_diagram.png")
plt.close()
