import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def get_transforms(split: str, img_size: int = 224):
    """Returns train or val/test transforms."""
    if split == 'train':
        return T.Compose([
            T.Resize((int(img_size*1.1), int(img_size*1.1))),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

class FractureDataset(Dataset):
    """Dataset for fracture images with optional bounding box cropping."""
    
    def __init__(self, df, img_root: str = '.', transform=None, use_bbox: bool = False):
        self.entries = df
        self.img_root = img_root
        self.transform = transform
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = row['image_path']
        
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)
        
        img = Image.open(img_path).convert('RGB')

        if self.use_bbox and all(k in row for k in ('bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax')):
            xmin = int(row['bbox_xmin'])
            ymin = int(row['bbox_ymin'])
            xmax = int(row['bbox_xmax'])
            ymax = int(row['bbox_ymax'])
            img = img.crop((xmin, ymin, xmax, ymax))

        label = int(row['label'])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_path
