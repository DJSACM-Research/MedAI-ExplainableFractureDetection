import torch
checkpoint_path = r"C:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\runs\classify\train\weights\vssm_base_0229_ckpt_epoch_237.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
sd = checkpoint.get('model', checkpoint)
keys = list(sd.keys())
print(f"Total keys: {len(keys)}")
# Print first 50
for k in keys[:50]:
    print(k)
# Print last 10
for k in keys[-10:]:
    print(k)
