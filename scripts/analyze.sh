#!/bin/bash
# Analysis wrapper script

python src/analysis/analyze.py \
    --checkpoint outputs/swin_mps/best.pth \
    --test-csv data/balanced_augmented_dataset/test.csv \
    --img-root data \
    --model swin \
    --class-names "Comminuted,Greenstick,Healthy,Oblique,Oblique Displaced,Spiral,Transverse,Transverse Displaced" \
    --out-dir outputs/analysis \
    "$@"
