#!/usr/bin/env python3
"""Debug what the Swin model is predicting to understand the label mismatch."""

import torch
import sys
sys.path.insert(0, '.')

from src.utils import get_model
import pandas as pd
from torch.utils.data import DataLoader
from src.utils import get_transforms, FractureDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load test data
test_df = pd.read_csv('data/balanced_augmented_dataset/test.csv').to_dict('records')
test_transform = get_transforms('test', 224)
test_dataset = FractureDataset(test_df, 'data', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Class names (the CORRECT mapping after our fix)
class_names = ['Comminuted', 'Greenstick', 'Healthy', 'Oblique_Displaced', 
               'Oblique', 'Spiral', 'Transverse_Displaced', 'Transverse']

print("="*70)
print("SWIN MODEL ANALYSIS")
print("="*70)

# Load Swin model
model = get_model('swin', num_classes=8, pretrained=False)
checkpoint = torch.load('models/best_swin.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Get predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print("True\\Pred", end="")
for i in range(8):
    print(f"{i:>5}", end="")
print()
for i in range(8):
    print(f"{i:>4}     ", end="")
    for j in range(8):
        print(f"{cm[i,j]:>5}", end="")
    print(f"  ({class_names[i]})")

# Find the mapping pattern
print("\n" + "="*70)
print("MAPPING ANALYSIS - What class does the model think each true class is?")
print("="*70)

for i in range(8):
    pred_for_class = cm[i]
    max_pred = pred_for_class.argmax()
    accuracy_for_class = cm[i,i] / pred_for_class.sum() * 100
    print(f"True class {i} ({class_names[i]:<20}): "
          f"predicted as class {max_pred} ({class_names[max_pred]:<20}) "
          f"{pred_for_class[max_pred]:>3}/{pred_for_class.sum()} times "
          f"(correct: {accuracy_for_class:.1f}%)")

# Check if there's a permutation pattern
print("\n" + "="*70)
print("HYPOTHESIS: Old model was trained with DIFFERENT label indices")
print("="*70)

# If we swap certain labels, what accuracy do we get?
print("\nTrying different label remappings...")

# The OLD incorrect mapping was:
# Oblique -> 3 (should be 4)
# Oblique_Displaced -> 4 (should be 3)
# Transverse -> 6 (should be 7)
# Transverse_Displaced -> 7 (should be 6)

# If the model was trained with old labels, we need to remap its predictions
old_to_new = {
    0: 0,  # Comminuted stays
    1: 1,  # Greenstick stays  
    2: 2,  # Healthy stays
    3: 4,  # Model's 3 (Oblique in old) -> 4 (Oblique in new)
    4: 3,  # Model's 4 (Oblique_Displaced in old) -> 3 (Oblique_Displaced in new)
    5: 5,  # Spiral stays
    6: 7,  # Model's 6 (Transverse in old) -> 7 (Transverse in new)
    7: 6,  # Model's 7 (Transverse_Displaced in old) -> 6 (Transverse_Displaced in new)
}

remapped_preds = [old_to_new[p] for p in all_preds]
remapped_accuracy = accuracy_score(all_labels, remapped_preds)

print(f"\nOriginal accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
print(f"Accuracy after remapping predictions: {remapped_accuracy*100:.2f}%")

if remapped_accuracy > 0.8:
    print("\n✅ CONFIRMED: The Swin model was trained with the OLD (incorrect) label mapping!")
    print("   When we remap its predictions to match our corrected labels, accuracy jumps!")
    print("\n   RECOMMENDATION: Retrain this model with the corrected dataset.")
else:
    print("\n❌ Remapping didn't help. The issue might be something else.")

# Also check hypercolumn model
print("\n" + "="*70)
print("HYPERCOLUMN_CBAM MODEL ANALYSIS")
print("="*70)

from src.models import HyperColumnCBAMDenseNet169
model2 = HyperColumnCBAMDenseNet169(num_classes=8, pretrained=False)
checkpoint2 = torch.load('models/best_hypercolumn_cbam_densenet169.pth', 
                         map_location='cpu', weights_only=False)
model2.load_state_dict(checkpoint2['model_state_dict'])
model2 = model2.to(device)
model2.eval()

all_preds2 = []
all_labels2 = []
with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model2(imgs)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_preds2.extend(preds.cpu().numpy().tolist())
        all_labels2.extend(labels.numpy().tolist())

remapped_preds2 = [old_to_new[p] for p in all_preds2]
remapped_accuracy2 = accuracy_score(all_labels2, remapped_preds2)

print(f"Original accuracy: {accuracy_score(all_labels2, all_preds2)*100:.2f}%")
print(f"Accuracy after remapping predictions: {remapped_accuracy2*100:.2f}%")

if remapped_accuracy2 > 0.8:
    print("\n✅ CONFIRMED: The HyperColumn model was ALSO trained with the OLD label mapping!")
    print("   RECOMMENDATION: Retrain this model with the corrected dataset.")
