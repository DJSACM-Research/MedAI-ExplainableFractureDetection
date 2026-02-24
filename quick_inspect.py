import torch
import os

checkpoint_path = r"C:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\runs\classify\train\weights\vssm_base_0229_ckpt_epoch_237.pth"
if not os.path.exists(checkpoint_path):
    print(f"File not found: {checkpoint_path}")
else:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Keys: {checkpoint.keys()}")
            if 'model' in checkpoint:
                sd = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            print("First 20 state_dict keys:")
            for k in list(sd.keys())[:20]:
                print(f"  {k}")
        else:
            print("Not a dictionary")
    except Exception as e:
        print(f"Error: {e}")
