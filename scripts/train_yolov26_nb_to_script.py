#!/usr/bin/env python
# coding: utf-8

# # YOLOv26 Fracture Classification Training
# 
# This notebook trains a YOLOv26 classification model on the `balanced_augmented_dataset`. 
# It utilizes the `ultralytics` library to load pretrained weights and fine-tune them on the fracture dataset.

# In[ ]:


# Install dependencies (if not already installed)
# Install dependencies (if not already installed)
# %pip install -U ultralytics
print("Ensure ultralytics is installed: pip install ultralytics")


# In[ ]:


import os
from ultralytics import YOLO
import torch

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# In[ ]:


# Define dataset path (absolute path recommended)
# The notebook is in 'notebooks/', so we go up one level to finding 'balanced_augmented_dataset'
# Define dataset path (absolute path recommended)
# We assume the script is run from the project root or adjust accordingly
try:
    # Try assuming we are in project root
    dataset_root = os.path.abspath("balanced_augmented_dataset")
    if not os.path.exists(dataset_root):
        # Maybe we are in scripts/
        dataset_root = os.path.abspath(os.path.join(os.getcwd(), "..", "balanced_augmented_dataset"))
except Exception:
    dataset_root = "c:/Users/hardi/OneDrive/Desktop/MedAIExplainableFractureDetection/balanced_augmented_dataset"

# Verify structure
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")
test_dir = os.path.join(dataset_root, "test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found at {train_dir}")

print(f"Dataset located at: {dataset_root}")


# In[ ]:


# Initialize YOLOv26 Classification Model
# We use 'yolov26n-cls.pt' (Nano) for efficiency. Automatically downloads pretrained weights.
model = YOLO('yolov26n-cls.pt') 

# Display model info
model.info()


# In[ ]:


# Train the model
# data: path to dataset containing train/ and val/ folders
# epochs: 20 (adjustable)
# imgsz: 224 (standard for classification)

results = model.train(
    data=dataset_root,
    epochs=20,
    imgsz=224,
    batch=16,
    device=device,
    project='../outputs/yolov26_training', # Save results to outputs folder
    name='fracture_cls'
)


# In[ ]:


# Validate on Validation Set
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")


# In[ ]:


# Predict on Test Set Images
# We can run prediction on the 'test' folder to see performance on unseen data
test_results = model.predict(source=test_dir, imgsz=224, save=True)

# Note: Predict saves results (labels/crops) but proper metric evaluation on the test set 
# usually requires the 'val()' mode pointing to the test split, or custom evaluation code.
# Here we'll rely on the visual outputs or manual metric calculation if needed.

print(f"Predictions saved to {test_results[0].save_dir}")


# In[ ]:


# Optional: Export to ONNX for deployment
path = model.export(format="onnx")
print(f"Model exported to {path}")

