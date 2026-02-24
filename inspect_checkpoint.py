import torch
import os

checkpoint_path = 'outputs/cross_validation/best_densenet169.pth'
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
else:
    try:
        ck = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ck.get('model_state_dict', ck)
        print("Keys found in checkpoint:")
        keys = list(state_dict.keys())
        # Print first 20 keys to get an idea
        for k in keys[:20]:
            print(k)
        # Check for specific cbam or hypercolumn keys
        cbam_keys = [k for k in keys if 'cbam' in k.lower() or 'attention' in k.lower()]
        if cbam_keys:
            print("\nCBAM/Attention keys found:")
            print(cbam_keys[:10])
        else:
            print("\nNo direct 'cbam' keys found.")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
