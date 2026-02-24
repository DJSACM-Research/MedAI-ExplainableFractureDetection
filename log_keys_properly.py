import torch
checkpoint_path = r"C:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection\runs\classify\train\weights\vssm_base_0229_ckpt_epoch_237.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
sd = checkpoint.get('model', checkpoint)
with open('ckpt_keys.txt', 'w') as f:
    for k in sd.keys():
        f.write(f"{k} {list(sd[k].shape)}\n")
print("Done")
