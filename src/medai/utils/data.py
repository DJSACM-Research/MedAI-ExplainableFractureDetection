"""
Data utilities for MedAI.

Provides dataset classes and data loading utilities.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "FractureDataset",
    "load_csv",
    "create_dataloader",
]


def load_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a CSV file into a list of dictionaries.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        List of dictionaries, one per row.
        
    Examples:
        >>> rows = load_csv('data/train.csv')
        >>> print(rows[0]['image_path'], rows[0]['label'])
    """
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


class FractureDataset(Dataset):
    """
    Dataset for fracture classification images.
    
    Supports:
    - Loading images from CSV with image_path and label columns
    - Optional bounding box cropping
    - Configurable transforms
    
    Args:
        data: List of dictionaries with 'image_path' and 'label' keys,
              or path to a CSV file.
        img_root: Root directory for resolving relative image paths.
        transform: Optional transform to apply to images.
        use_bbox: Whether to crop using bounding box columns if available.
        
    Examples:
        >>> dataset = FractureDataset('train.csv', transform=get_transforms('train'))
        >>> img, label, path = dataset[0]
    """
    
    def __init__(
        self,
        data: Union[str, Path, List[Dict[str, Any]]],
        img_root: str = ".",
        transform: Optional[Callable] = None,
        use_bbox: bool = False,
    ):
        # Load data from CSV if path is provided
        if isinstance(data, (str, Path)):
            self.entries = load_csv(data)
        else:
            self.entries = data
            
        self.img_root = Path(img_root)
        self.transform = transform
        self.use_bbox = use_bbox
        
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single item from the dataset.
        
        Returns:
            Tuple of (image_tensor, label, image_path)
        """
        row = self.entries[idx]
        img_path = row["image_path"]
        
        # Resolve relative paths
        if not os.path.isabs(img_path):
            img_path = str(self.img_root / img_path)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Optional bounding box cropping
        if self.use_bbox and self._has_bbox(row):
            img = self._crop_bbox(img, row)
        
        # Get label
        label = int(row["label"])
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_path
    
    def _has_bbox(self, row: Dict[str, Any]) -> bool:
        """Check if row has valid bounding box columns."""
        bbox_keys = ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
        return all(k in row and row[k] for k in bbox_keys)
    
    def _crop_bbox(self, img: Image.Image, row: Dict[str, Any]) -> Image.Image:
        """Crop image to bounding box."""
        xmin = int(float(row["bbox_xmin"]))
        ymin = int(float(row["bbox_ymin"]))
        xmax = int(float(row["bbox_xmax"]))
        ymax = int(float(row["bbox_ymax"]))
        return img.crop((xmin, ymin, xmax, ymax))
    
    def get_labels(self) -> List[int]:
        """Return all labels in the dataset."""
        return [int(row["label"]) for row in self.entries]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Return the distribution of classes in the dataset."""
        from collections import Counter
        return dict(Counter(self.get_labels()))


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults.
    
    Args:
        dataset: The dataset to load.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
