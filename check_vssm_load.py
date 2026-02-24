import torch
import torch.nn as nn
from vssm_stub import VSSM

# Refine OP implementation to match keys
class SS2D_Keys(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_inner = d_model * 2
        dt_rank = math.ceil(d_model / 16)
        self.x_proj_weight = nn.Parameter(torch.zeros(4, dt_rank + 2, d_inner)) # Adjusting 10 to (8+2)
        self.dt_projs_weight = nn.Parameter(torch.zeros(4, d_inner, dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.zeros(4, d_inner))
        self.A_logs = nn.Parameter(torch.zeros(d_inner * 4, 1))
        self.Ds = nn.Parameter(torch.zeros(d_inner * 4))
        self.out_norm = nn.LayerNorm(d_inner)
        self.in_proj = nn.Linear(d_model, d_inner)
        self.conv2d = nn.Conv2d(d_inner, d_inner, 3, groups=d_inner, padding=1)
        self.out_proj = nn.Linear(d_inner, d_model)

import math

def check_load():
    model = VSSM()
    # Override blocks.op to match keys
    for i, layer in enumerate(model.layers):
        dim = model.layers[i].blocks[0].norm.normalized_shape[0]
        for b in range(len(layer.blocks)):
            layer.blocks[b].op = SS2D_Keys(dim)
    
    weights_path = r"C:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\runs\classify\train\weights\vssm_base_0229_ckpt_epoch_237.pth"
    checkpoint = torch.load(weights_path, map_location='cpu')
    sd = checkpoint.get('model', checkpoint)
    
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    print(f"Missing: {len(missing)}")
    print(f"Unexpected: {len(unexpected)}")
    
    if len(missing) < 50:
        print("First 20 missing:")
        for m in missing[:20]:
            print(f"  {m}")
    else:
        print(f"Too many missing: {len(missing)}")
        
    print(f"First 20 unexpected:")
    for u in unexpected[:20]:
        print(f"  {u}")

if __name__ == "__main__":
    check_load()
