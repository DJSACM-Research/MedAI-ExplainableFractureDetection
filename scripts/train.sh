#!/bin/bash
# Training wrapper script

python src/training/pipeline.py \
    --train-csv data/balanced_augmented_dataset/train.csv \
    --val-csv data/balanced_augmented_dataset/val.csv \
    --test-csv data/balanced_augmented_dataset/test.csv \
    --img-root data \
    --model swin \
    --num-classes 8 \
    --epochs 20 \
    --batch-size 6 \
    --out-dir outputs/swin_mps \
    --wandb-project fracture-detection \
    "$@"
