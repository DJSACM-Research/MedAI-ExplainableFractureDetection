import torch.nn as nn
import timm
import torchvision.models as tvmodels

def get_model(name: str, num_classes: int, pretrained: bool = True):
    """Loads and adapts model architecture."""
    name = name.lower()
    
    if name.startswith('swin'):
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    
    if name.startswith('convnext'):
        model = timm.create_model('convnext_tiny', pretrained=pretrained)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        else:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
    
    if name.startswith('densenet'):
        model = tvmodels.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    
    raise ValueError(f'Unknown model: {name}')
