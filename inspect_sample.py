import numpy as np
import os

gt_path = 'dataset/train/train/GT/000000.npy'
noisy_path = 'dataset/train/train/NoisyLR/000000.npy'

gt_arr = np.load(gt_path)
noisy_arr = np.load(noisy_path)

print(f"GT shape: {gt_arr.shape}")
print(f"GT dtype: {gt_arr.dtype}")
print(f"GT min: {gt_arr.min()}, GT max: {gt_arr.max()}")

print(f"NoisyLR shape: {noisy_arr.shape}")
print(f"NoisyLR dtype: {noisy_arr.dtype}")
print(f"NoisyLR min: {noisy_arr.min()}, NoisyLR max: {noisy_arr.max()}")
