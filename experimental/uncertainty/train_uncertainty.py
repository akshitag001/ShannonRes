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
from torch.cuda.amp import GradScaler, autocast

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
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--run_name", type=str, default="", help="Suffix for saving logs and weights (e.g. 'seed2')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--finetune_uncertainty", action="store_true", help="Fine-tune the uncertainty branch")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs

    if args.synthesize_prob is not None:
        config['synthesize_prob'] = args.synthesize_prob
        
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    if args.finetune_uncertainty:
        args.run_name = "uncertainty"
        if args.resume is None:
            print("Warning: --finetune_uncertainty normally expects --resume to load the backbone weights.")

    if args.smoke_test:
        config['num_epochs'] = 2
        config['batch_size'] = 4
        print("SMOKE TEST MODE ENABLED: Limiting to 2 epochs, batch size 4")
        
    if args.finetune_uncertainty and not args.smoke_test:
        config['num_epochs'] = config.get('warmup_epochs', 5) + config.get('hetero_epochs', 15)

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn.benchmark for fixed-size inputs")
    
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
                noisy = noisy.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                
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
    if args.finetune_uncertainty:
        current_lr = config.get('warmup_lr', 0.0001)
    else:
        current_lr = config['learning_rate']
        
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    criterion = RestorationLoss(
        lpips_model=lpips_model,
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_ssim=config.get('lambda_ssim', 0.5),
        lambda_lpips=config.get('lambda_lpips', 0.5),
        lambda_hetero=config.get('lambda_hetero', 0.0),
        device=device
    )
    
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1
            if args.finetune_uncertainty:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                start_epoch = 0 # Restart epoch count for fine-tuning
                best_psnr = 0.0
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_psnr = checkpoint.get('val_psnr', 0.0)
            print(f"Resuming training from epoch {start_epoch} with previous best PSNR: {best_psnr:.4f}")
        else:
            print(f"Error: No checkpoint found at '{args.resume}'")
            return

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Initialize train logs
    suffix = f"_{args.run_name}" if args.run_name else ""
    metrics_log_file = os.path.join(config['save_dir'], f'train_log{suffix}.csv')
    audit_log_file = os.path.join(config['save_dir'], f'synthesis_audit{suffix}.csv')
    
    mode = 'a' if args.resume else 'w'
    
    with open(metrics_log_file, mode, newline='') as f:
        writer = csv.writer(f)
        if not args.resume or (args.finetune_uncertainty and start_epoch == 0):
            if args.finetune_uncertainty:
                writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'val_sigma', 'seconds'])
            else:
                writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'seconds'])
        
    with open(audit_log_file, mode, newline='') as f:
        writer = csv.writer(f)
        if not args.resume:
            writer.writerow(['epoch', 'filename', 'is_synthetic'])

    # Initialize AMP Scaler
    scaler = GradScaler()

    # Training Loop
    if args.finetune_uncertainty:
        model.predict_uncertainty = True

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        is_warmup = False
        # Freezing logic
        if args.finetune_uncertainty:
            warmup_epochs = config.get('warmup_epochs', 5)
            is_warmup = epoch < warmup_epochs
            
            if epoch == warmup_epochs:
                print(f"Switching to Phase B: Heteroscedastic Fine-tuning with LR={config['learning_rate']}")
                for g in optimizer.param_groups:
                    g['lr'] = config['learning_rate']
                    
            for name, param in model.named_parameters():
                if 'conv_uncertainty' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for noisy, gt, filenames, is_synthetic in pbar:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with autocast(dtype=amp_dtype):
                if args.finetune_uncertainty:
                    pred, sigma = model(noisy)
                    if is_warmup:
                        with torch.no_grad():
                            err_map = torch.abs(pred - gt)
                        loss = F.mse_loss(sigma, err_map)
                        l1 = 0.0; ssim = 0.0; lpips_val = 0.0
                    else:
                        loss, l1, ssim, lpips_val = criterion(pred, gt, sigma=sigma)
                else:
                    pred = model(noisy)
                    loss, l1, ssim, lpips_val = criterion(pred, gt)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        val_psnr, val_ssim, val_lpips, val_sigma = 0.0, 0.0, 0.0, 0.0
        all_sigmas = []
        all_errors = []
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for noisy, gt, filenames, is_synthetic in pbar_val:
                noisy = noisy.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                
                if args.finetune_uncertainty:
                    pred, sigma = model(noisy)
                    val_sigma += sigma.mean().item()
                    if is_warmup:
                        err_map = torch.abs(pred - gt)
                        all_sigmas.append(sigma.cpu().numpy().flatten())
                        all_errors.append(err_map.cpu().numpy().flatten())
                else:
                    pred = model(noisy)
                    
                psnr, ssim, lpips_val = evaluator.evaluate(pred, gt)
                
                val_psnr += psnr
                val_ssim += ssim
                val_lpips += lpips_val
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        if args.finetune_uncertainty:
            val_sigma /= len(val_loader)
            if is_warmup and len(all_sigmas) > 0:
                import numpy as np
                from scipy.stats import pearsonr
                sf = np.concatenate(all_sigmas)
                ef = np.concatenate(all_errors)
                corr, _ = pearsonr(sf, ef)
                print(f"Epoch {epoch+1} Warmup Correlation (Sigma vs L1 Error): {corr:.4f}")
        
        epoch_duration = time.time() - epoch_start_time
        
        # Log metrics to metrics log
        with open(metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if args.finetune_uncertainty:
                writer.writerow([epoch+1, avg_train_loss, val_psnr, val_ssim, val_lpips, val_sigma, epoch_duration])
            else:
                writer.writerow([epoch+1, avg_train_loss, val_psnr, val_ssim, val_lpips, epoch_duration])
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} | Time: {epoch_duration:.1f}s")
        
        # Save checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            suffix = f"_{args.run_name}" if args.run_name else ""
            save_path = os.path.join(config['save_dir'], f'best_model{suffix}.pth')
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
