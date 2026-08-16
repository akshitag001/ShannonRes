import torch
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import yaml

from src.dataset import ImageRestorationDataset
from src.model import RestorationCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = "weights/best_model_uncertainty.pth"
    
    print("Loading checkpoint...")
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
    
    all_sigma = []
    all_l1_error = []
    all_gt = []
    
    print("Evaluating validation set...")
    with torch.no_grad():
        # Evaluate a subset of 10-20 images to be fast, but enough for a good statistical sample
        max_images = 20
        num_images = min(len(val_ds), max_images)
        for i in tqdm(range(num_images)):
            noisy, gt, _, _ = val_ds[i]
            
            noisy_t = noisy.unsqueeze(0).to(device)
            gt_t = gt.unsqueeze(0).to(device)
            
            restored, sigma = model(noisy_t)
            
            l1_error = torch.abs(restored - gt_t)
            
            all_sigma.append(sigma.squeeze().cpu().numpy().flatten())
            all_l1_error.append(l1_error.squeeze().cpu().numpy().flatten())
            all_gt.append(gt.squeeze().cpu().numpy().flatten())
            
    # Concatenate all arrays
    sigma_flat = np.concatenate(all_sigma)
    error_flat = np.concatenate(all_l1_error)
    gt_flat = np.concatenate(all_gt)
    
    print("\nComputing Pearson Correlations...")
    corr_error, _ = pearsonr(sigma_flat, error_flat)
    corr_brightness, _ = pearsonr(sigma_flat, gt_flat)
    
    print(f"\nResults:")
    print(f"Correlation (Sigma vs Actual L1 Error): {corr_error:.4f}")
    print(f"Correlation (Sigma vs GT Brightness):  {corr_brightness:.4f}")
    
    if corr_brightness > corr_error:
        print("\nWARNING: Sigma correlates more strongly with image brightness than with actual error!")
        print("The uncertainty head may have learned a 'brightness shortcut'.")
    else:
        print("\nSUCCESS: Sigma correlates primarily with actual error. The uncertainty head is functioning as intended.")

if __name__ == "__main__":
    main()
