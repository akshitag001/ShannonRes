import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import ImageRestorationDataset
from src.model import RestorationCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = "weights/best_model_uncertainty.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found. Please train the uncertainty head first.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = RestorationCNN(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_features=config['num_features'],
        num_res_blocks=config['num_res_blocks'],
        scale=checkpoint['scale']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.predict_uncertainty = True
    model.eval()
    
    val_ds = ImageRestorationDataset(
        gt_dir=config['train_gt_dir'],
        noisy_dir=config['train_noisy_dir'],
        gt_crop_size=config['gt_crop_size'],
        synthesize_prob=0.0
    )
    
    # Select a few indices, we can just take the first 4
    indices = [0, 1, 2, 3] 
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    
    if len(indices) == 1:
        axes = [axes]
        
    with torch.no_grad():
        for i, idx in enumerate(indices):
            noisy, gt, filename, _ = val_ds[idx]
            noisy_t = noisy.unsqueeze(0).to(device)
            
            restored, sigma = model(noisy_t)
            
            noisy_img = noisy.squeeze(0).cpu().numpy()
            restored_img = restored.squeeze().cpu().numpy()
            sigma_img = sigma.squeeze().cpu().numpy()
            
            axes[i][0].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
            axes[i][0].set_title(f"Noisy Input\n{filename}")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(restored_img, cmap='gray', vmin=0, vmax=1)
            axes[i][1].set_title("Restored Output")
            axes[i][1].axis('off')
            
            # Normalize sigma for heatmap
            sigma_norm = (sigma_img - sigma_img.min()) / (sigma_img.max() - sigma_img.min() + 1e-8)
            im = axes[i][2].imshow(sigma_norm, cmap='inferno')
            axes[i][2].set_title("Uncertainty Heatmap")
            axes[i][2].axis('off')
            plt.colorbar(im, ax=axes[i][2], fraction=0.046, pad=0.04)
            
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = "results/uncertainty_examples.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved qualitative examples to {out_path}")

if __name__ == "__main__":
    main()
