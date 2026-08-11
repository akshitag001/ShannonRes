import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import random
import numpy as np
import csv
import time
import lpips
import torch.nn.functional as F

from src.dataset import ImageRestorationDataset
from src.model import RestorationCNN
from src.losses import RestorationLoss
from src.metrics import MetricsEvaluator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Image Restoration Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--synthesize_prob", type=float, default=None, help="Override synthetic noise probability")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick 2-epoch test on 32 samples")
    parser.add_argument("--baseline_only", action="store_true", help="Evaluate Bicubic baseline on validation set and exit")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.synthesize_prob is not None:
        config['synthesize_prob'] = args.synthesize_prob

    if args.smoke_test:
        config['num_epochs'] = 2
        config['batch_size'] = 4
        print("SMOKE TEST MODE ENABLED: Limiting to 2 epochs, batch size 4")

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading LPIPS VGG model...")
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # Dataset Creation - Separate train and val to enforce synthesize_prob=0 on val
    train_full_ds = ImageRestorationDataset(
        gt_dir=config['train_gt_dir'],
        noisy_dir=config['train_noisy_dir'],
        gt_crop_size=config['gt_crop_size'],
        synthesize_prob=config.get('synthesize_prob', 0.5)
    )
    
    val_full_ds = ImageRestorationDataset(
        gt_dir=config['train_gt_dir'],
        noisy_dir=config['train_noisy_dir'],
        gt_crop_size=config['gt_crop_size'],
        synthesize_prob=0.0  # Force 0 for validation
    )
    
    total_len = len(train_full_ds)
    if args.smoke_test:
        total_len = min(32, total_len)
        
    indices = list(range(total_len))
    random.shuffle(indices)
    
    val_size = int(total_len * config['val_split_ratio'])
    if val_size == 0 and total_len > 1:
        val_size = 1
    train_size = total_len - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(train_full_ds, train_indices)
    val_dataset = Subset(val_full_ds, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 0), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 0), pin_memory=True)

    evaluator = MetricsEvaluator(lpips_model=lpips_model, device=device)

    if args.baseline_only:
        print("Running Baseline Only (Bicubic Upsampling) on Validation Set...")
        val_psnr, val_ssim, val_lpips = 0.0, 0.0, 0.0
        with torch.no_grad():
            for noisy, gt, filenames, is_synthetic in tqdm(val_loader, desc="Baseline"):
                noisy = noisy.to(device)
                gt = gt.to(device)
                
                # Baseline prediction
                pred = F.interpolate(noisy, scale_factor=val_full_ds.scale, mode='bicubic', align_corners=False)
                pred = torch.clamp(pred, 0.0, 1.0)
                
                psnr, ssim, lpips_val = evaluator.evaluate(pred, gt)
                val_psnr += psnr
                val_ssim += ssim
                val_lpips += lpips_val
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        print(f"Baseline Results | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f}")
        return

    # Model
    model = RestorationCNN(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_features=config['num_features'],
        num_res_blocks=config['num_res_blocks'],
        scale=train_full_ds.scale
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = RestorationLoss(
        lpips_model=lpips_model,
        lambda_l1=config['lambda_l1'],
        lambda_ssim=config['lambda_ssim'],
        lambda_lpips=config['lambda_lpips'],
        device=device
    )

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Initialize train logs
    metrics_log_file = os.path.join(config['save_dir'], 'train_log.csv')
    audit_log_file = os.path.join(config['save_dir'], 'synthesis_audit.csv')
    
    with open(metrics_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'seconds'])
        
    with open(audit_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'filename', 'is_synthetic'])

    best_psnr = 0.0

    # Training Loop
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for noisy, gt, filenames, is_synthetic in pbar:
            noisy = noisy.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            pred = model(noisy)
            loss, l1, ssim, lpips_val = criterion(pred, gt)
            
            loss.backward()
            optimizer.step()
            
            # Log sample source to audit log
            with open(audit_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for fname, is_syn in zip(filenames, is_synthetic):
                    writer.writerow([epoch+1, fname, is_syn.item() if torch.is_tensor(is_syn) else is_syn])
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1': f"{l1:.4f}", 'SSIM': f"{ssim:.4f}", 'LPIPS': f"{lpips_val:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_psnr, val_ssim, val_lpips = 0.0, 0.0, 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for noisy, gt, filenames, is_synthetic in pbar_val:
                noisy = noisy.to(device)
                gt = gt.to(device)
                
                pred = model(noisy)
                psnr, ssim, lpips_val = evaluator.evaluate(pred, gt)
                
                val_psnr += psnr
                val_ssim += ssim
                val_lpips += lpips_val
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        
        epoch_duration = time.time() - epoch_start_time
        
        # Log metrics to metrics log
        with open(metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, val_psnr, val_ssim, val_lpips, epoch_duration])
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} | Time: {epoch_duration:.1f}s")
        
        # Save checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'scale': train_full_ds.scale,
                'config': config
            }, save_path)
            print(f"Saved new best model to {save_path}")

if __name__ == "__main__":
    main()
