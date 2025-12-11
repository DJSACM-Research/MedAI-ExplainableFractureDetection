import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from app import get_model

IMG_PATH = './test_images/Spiral_257_jpg.rf.3cc9912ab33e60062d99c277a5aa9bf7_0010.jpg'
CLASS_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Oblique', 'Oblique Displaced', 'Spiral', 'Transverse', 'Transverse Displaced']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert('RGB')
x = transform(img).unsqueeze(0)

print(f'Testing with: {IMG_PATH}')
print(f'Expected class: Spiral')
print('='*70)

# Test all hypercolumn models
hypercolumn_models = [
    'hypercolumn_cbam_densenet169',
    'hypercolumn_cbam_densenet169_focal', 
    'hypercolumn_densenet169',
    'hypercolumn_densenet169_old'
]

for model_name in hypercolumn_models:
    try:
        model = get_model(model_name, num_classes=8)
        cp = torch.load(f'./updated_models/best_{model_name}.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(cp.get('model_state_dict', cp))
        model.eval()
        
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output, dim=1)[0]
            pred = probs.argmax().item()
            print(f'{model_name}: {CLASS_NAMES[pred]} ({probs[pred].item()*100:.1f}%)')
    except Exception as e:
        print(f'{model_name}: ERROR - {e}')

print('='*70)
print('Done!')
