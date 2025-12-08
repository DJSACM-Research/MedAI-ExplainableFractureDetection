import os, sys, csv, argparse, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import get_device, get_model, get_transforms

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader]

def save_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i, str(cm[i,j]), ha='center', va='center', color='black')
    plt.colorbar(im)
    plt.tight_layout(); plt.savefig(out_path); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    # Allow multiple models and checkpoints
    parser.add_argument('--model', action='append', help='Model architecture name', required=True)
    parser.add_argument('--checkpoint', action='append', help='Path to checkpoint', required=True)
    
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--img-root', default='.')
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('--class-names', required=True)
    parser.add_argument('--out-dir', default='outputs/analysis/ensemble')
    args = parser.parse_args()
    
    if len(args.model) != len(args.checkpoint):
        print("Error: Number of models must match number of checkpoints")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    class_names = [s.strip() for s in args.class_names.split(',')]
    num_classes = len(class_names)
    device = get_device()

    # Load all models
    models = []
    print(f"Loading {len(args.model)} models for ensemble...")
    for m_name, ck_path in zip(args.model, args.checkpoint):
        print(f"  - Loading {m_name} from {ck_path}")
        model = get_model(m_name, num_classes, pretrained=False)
        try:
            ck = torch.load(ck_path, map_location='cpu')
            if isinstance(ck, dict) and 'model_state_dict' in ck:
                model.load_state_dict(ck['model_state_dict'])
            else:
                model.load_state_dict(ck)
            model.to(device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"  ❌ Failed to load {m_name}: {e}")
            sys.exit(1)

    rows = load_csv(args.test_csv)
    tf = get_transforms('val', args.img_size)
    preds, trues, paths = [], [], []
    
    print(f"Running inference on {len(rows)} images...")
    
    for i, r in enumerate(rows):
        img_path = r['image_path'] if os.path.isabs(r['image_path']) else os.path.join(args.img_root, r['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            t = tf(img).unsqueeze(0).to(device)
            
            ensemble_prob = None
            
            with torch.no_grad():
                for model in models:
                    out = model(t)
                    prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                    if ensemble_prob is None:
                        ensemble_prob = prob
                    else:
                        ensemble_prob += prob
            
            # Average probabilities
            ensemble_prob /= len(models)
            pred = int(ensemble_prob.argmax())
            
            preds.append(pred)
            trues.append(int(r['label']))
            paths.append(img_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    cm = confusion_matrix(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average=None, labels=list(range(num_classes)), zero_division=0)

    # print per-class metrics
    print("\nEnsemble Results:")
    for i,name in enumerate(class_names):
        print(f'{i} {name}: support={(cm[i].sum())}, prec={p[i]:.3f}, rec={r[i]:.3f}, f1={f1[i]:.3f}')
    print('macro-f1:', np.mean(f1))

    # save confusion matrix image
    save_confusion(cm, class_names, os.path.join(args.out_dir,'confusion_matrix.png'))
    print(f'Saved results to {args.out_dir}')

if __name__=='__main__':
    main()
