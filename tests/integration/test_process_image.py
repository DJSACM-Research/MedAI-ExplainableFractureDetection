import os
import torch
import torch.nn as nn
from PIL import Image

import pytest

from backend_hf import app as medai_app_module


class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # return logits with strong prediction for class 0
        logits = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)
        logits[:, 0] = 10.0
        return logits


def test_process_image_basic():
    # ensure models dict can accept a dummy
    from backend_hf.app import process_image, models, device, CLASS_NAMES

    # Inject dummy model
    dummy = DummyModel(len(CLASS_NAMES))
    dummy.to(device)
    models['dummy_test_model'] = dummy

    # Create a trivial image
    img = Image.new('RGB', (224, 224), color='white')

    result = process_image(img, use_conformal='false', ensemble_mode='weighted', stacker_path=None)

    assert 'prediction' in result
    assert 'ensemble' in result
    assert result['prediction']['top_class'] in CLASS_NAMES
