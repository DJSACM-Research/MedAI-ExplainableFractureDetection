#!/usr/bin/env python3
"""Investigate label mapping across train/val/test splits."""

import pandas as pd

train_df = pd.read_csv('data/balanced_augmented_dataset/train.csv')
val_df = pd.read_csv('data/balanced_augmented_dataset/val.csv')
test_df = pd.read_csv('data/balanced_augmented_dataset/test.csv')

print("="*70)
print("LABEL MAPPING INVESTIGATION")
print("="*70)

# Extract class names from paths
def get_class_from_path(path):
    parts = path.split('/')
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"

train_df['class_name'] = train_df['image_path'].apply(get_class_from_path)
val_df['class_name'] = val_df['image_path'].apply(get_class_from_path)
test_df['class_name'] = test_df['image_path'].apply(get_class_from_path)

print("\n1. TRAIN SET - Label to Class Mapping:")
train_mapping = train_df.groupby(['label', 'class_name']).size().reset_index(name='count')
print(train_mapping.to_string(index=False))

print("\n2. VAL SET - Label to Class Mapping:")
val_mapping = val_df.groupby(['label', 'class_name']).size().reset_index(name='count')
print(val_mapping.to_string(index=False))

print("\n3. TEST SET - Label to Class Mapping:")
test_mapping = test_df.groupby(['label', 'class_name']).size().reset_index(name='count')
print(test_mapping.to_string(index=False))

# Check if mappings are consistent
print("\n4. MAPPING COMPARISON (Class -> Label):")
train_class_to_label = dict(zip(train_df['class_name'], train_df['label']))
val_class_to_label = dict(zip(val_df['class_name'], val_df['label']))
test_class_to_label = dict(zip(test_df['class_name'], test_df['label']))

all_classes = sorted(set(train_class_to_label.keys()) | set(val_class_to_label.keys()) | set(test_class_to_label.keys()))

print(f"{'Class Name':<25} {'Train':<8} {'Val':<8} {'Test':<8} {'Status'}")
print("-"*60)
mismatches = []
for cls in all_classes:
    train_lbl = train_class_to_label.get(cls, 'N/A')
    val_lbl = val_class_to_label.get(cls, 'N/A')
    test_lbl = test_class_to_label.get(cls, 'N/A')
    if train_lbl == val_lbl == test_lbl:
        status = "OK"
    else:
        status = "MISMATCH!"
        mismatches.append(cls)
    print(f"{cls:<25} {str(train_lbl):<8} {str(val_lbl):<8} {str(test_lbl):<8} {status}")

if mismatches:
    print(f"\n*** FOUND {len(mismatches)} LABEL MISMATCHES! ***")
else:
    print("\n*** All labels are consistent across splits ***")
