import torch
from vssm import vssm_base

def check_load():
    model = vssm_base()
    weights_path = r"C:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\runs\classify\train\weights\vssm_base_0229_ckpt_epoch_237.pth"
    checkpoint = torch.load(weights_path, map_location='cpu')
    sd = checkpoint.get('model', checkpoint)
    
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    print(f"Missing (Total {len(missing)}):")
    if missing:
        for m in missing[:10]:
            print(f"  {m}")
            
    print(f"Unexpected (Total {len(unexpected)}):")
    if unexpected:
        for u in unexpected[:10]:
            print(f"  {u}")

    if len(missing) == 0 and len(unexpected) == 0:
         print("✅ PERFECT MATCH!")
    elif len(missing) == 0:
         print("✅ All model weights found in checkpoint (with some extra in ckpt).")

if __name__ == "__main__":
    check_load()
