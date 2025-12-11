#!/usr/bin/env python3
"""
Fix the label mapping mismatch in test.csv.

The test.csv has incorrect labels for 4 classes:
- Oblique: should be 4 (has 3)
- Oblique_Displaced: should be 3 (has 4)
- Transverse: should be 7 (has 6)
- Transverse_Displaced: should be 6 (has 7)

This script fixes the labels to match train/val mapping.
"""

import pandas as pd
import shutil
from datetime import datetime

# Load test CSV
test_path = 'data/balanced_augmented_dataset/test.csv'
test_df = pd.read_csv(test_path)

# Backup original
backup_path = f'data/balanced_augmented_dataset/test_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
shutil.copy(test_path, backup_path)
print(f"Backed up original to: {backup_path}")

# Extract class names from paths
def get_class_from_path(path):
    parts = path.split('/')
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"

test_df['class_name'] = test_df['image_path'].apply(get_class_from_path)

# Correct label mapping (from train/val)
correct_mapping = {
    'Comminuted': 0,
    'Greenstick': 1,
    'Healthy': 2,
    'Oblique_Displaced': 3,
    'Oblique': 4,
    'Spiral': 5,
    'Transverse_Displaced': 6,
    'Transverse': 7
}

print("\nBefore fix:")
print(test_df.groupby(['label', 'class_name']).size().reset_index(name='count').to_string(index=False))

# Fix labels based on class name
test_df['label'] = test_df['class_name'].map(correct_mapping)

print("\nAfter fix:")
print(test_df.groupby(['label', 'class_name']).size().reset_index(name='count').to_string(index=False))

# Remove helper column and save
test_df = test_df.drop(columns=['class_name'])
test_df.to_csv(test_path, index=False)

print(f"\n✓ Fixed test.csv saved to: {test_path}")
print(f"\nLabel distribution after fix:")
print(test_df['label'].value_counts().sort_index())
