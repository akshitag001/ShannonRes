import os
import numpy as np
import matplotlib.pyplot as plt

def extract_sample(index):
    gt_path = f"dataset/train/train/GT/{index:06d}.npy"
    noisy_path = f"dataset/train/train/NoisyLR/{index:06d}.npy"
    
    if os.path.exists(gt_path) and os.path.exists(noisy_path):
        gt_data = np.load(gt_path)
        noisy_data = np.load(noisy_path)
        
        plt.imsave(f"sample_gt_{index:06d}.png", gt_data, cmap='gray')
        plt.imsave(f"sample_noisy_{index:06d}.png", noisy_data, cmap='gray')
        print(f"Extracted {index:06d}")
    else:
        print(f"Missing files for {index:06d}")

for i in range(3):
    extract_sample(i)
