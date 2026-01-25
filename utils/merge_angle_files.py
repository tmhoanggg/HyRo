# Run this after inference is finished
import torch
import glob

rank_files = sorted(glob.glob('angle_statistics_rank*.pt'))
all_angles = []

for file in rank_files:
    data = torch.load(file, weights_only=False)

    # data['angles'] is a list of tensors
    # Stack per-rank first
    rank_angles = torch.stack(data['angles'], dim=0)  # [N_samples_rank, ...]
    all_angles.append(rank_angles)

# Concatenate across ranks
merged_angles = torch.cat(all_angles, dim=0)

torch.save({'angles': merged_angles}, 'angle_statistics_final.pt')
print(f"Merged shape: {merged_angles.shape}")
