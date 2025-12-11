#!/usr/bin/env python3
"""Analyze data split to understand train/val vs test accuracy gap."""

import pandas as pd

train_df = pd.read_csv('data/balanced_augmented_dataset/train.csv')
val_df = pd.read_csv('data/balanced_augmented_dataset/val.csv')
test_df = pd.read_csv('data/balanced_augmented_dataset/test.csv')

print("="*60)
print("DATA SPLIT ANALYSIS")
print("="*60)

# Check source types
print('\n1. IMAGE TYPES:')
train_aug = train_df[train_df['image_path'].str.contains('aug_')]
val_aug = val_df[val_df['image_path'].str.contains('aug_')]
test_aug = test_df[test_df['image_path'].str.contains('aug_')]

print(f'TRAIN: {len(train_aug)} augmented ({100*len(train_aug)/len(train_df):.1f}%), {len(train_df) - len(train_aug)} original')
print(f'VAL:   {len(val_aug)} augmented ({100*len(val_aug)/len(val_df):.1f}%), {len(val_df) - len(val_aug)} original')
print(f'TEST:  {len(test_aug)} augmented ({100*len(test_aug)/len(test_df):.1f}%), {len(test_df) - len(test_aug)} original')

print('\n2. LABEL DISTRIBUTION COMPARISON:')
print(f"{'Label':<8} {'Train %':<12} {'Val %':<12} {'Test %':<12}")
print("-"*44)
for label in range(8):
    train_pct = 100 * (train_df['label'] == label).sum() / len(train_df)
    val_pct = 100 * (val_df['label'] == label).sum() / len(val_df)
    test_pct = 100 * (test_df['label'] == label).sum() / len(test_df)
    print(f'{label:<8} {train_pct:<12.1f} {val_pct:<12.1f} {test_pct:<12.1f}')

print('\n3. SAMPLE SIZES:')
print(f'Train: {len(train_df)}')
print(f'Val:   {len(val_df)} ({100*len(val_df)/(len(train_df)+len(val_df)+len(test_df)):.1f}%)')
print(f'Test:  {len(test_df)} ({100*len(test_df)/(len(train_df)+len(val_df)+len(test_df)):.1f}%)')

# Check if val is from augmented source same as train
print('\n4. PATH STRUCTURE:')
print(f"Train example: {train_df['image_path'].iloc[0]}")
print(f"Val example:   {val_df['image_path'].iloc[0]}")
print(f"Test example:  {test_df['image_path'].iloc[0]}")
