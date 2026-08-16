import torch
import numpy as np
import lpips
from torch.utils.data import DataLoader, Subset
from src.dataset import ImageRestorationDataset
from src.model import RestorationCNN
from src.metrics import MetricsEvaluator
import argparse
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    args = parser.parse_args()

    device = 'cuda'
    
    models = []
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get('config', {
            'in_channels': 1, 'out_channels': 1, 'num_features': 64, 'num_res_blocks': 16,
            'train_gt_dir': 'dataset/train/train/GT', 'train_noisy_dir': 'dataset/train/train/NoisyLR',
            'gt_crop_size': 128, 'val_split_ratio': 0.1, 'seed': 1234
        })
        config['train_gt_dir'] = 'dataset/train/train/GT'
        config['train_noisy_dir'] = 'dataset/train/train/NoisyLR'
        
        model = RestorationCNN(config['in_channels'], config['out_channels'], config['num_features'], config['num_res_blocks'], 2).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        models.append(model)
        
    set_seed(config['seed'])

    lpips_net = lpips.LPIPS(net='vgg').to(device)
    evaluator = MetricsEvaluator(lpips_net, device)

    val_ds = ImageRestorationDataset(config['train_gt_dir'], config['train_noisy_dir'], gt_crop_size=config['gt_crop_size'], synthesize_prob=0.0)
    
    indices = list(range(len(val_ds)))
    random.shuffle(indices)
    val_size = int(len(val_ds) * config['val_split_ratio'])
    val_idx = indices[-val_size:]
    val_sub = Subset(val_ds, val_idx)
    loader = DataLoader(val_sub, batch_size=4)

    psnr_t, ssim_t, lpips_t = 0, 0, 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            
            ensemble_p = []
            for m in models:
                p = m(x)
                if isinstance(p, tuple): p = p[0]
                ensemble_p.append(p)
            
            p = torch.mean(torch.stack(ensemble_p), dim=0)
            ps, ss, lp = evaluator.evaluate(p, y)
            psnr_t += ps
            ssim_t += ss
            lpips_t += lp

    print(f"Metrics for Ensemble {args.checkpoints}:")
    print(f"PSNR: {psnr_t/len(loader):.4f} | SSIM: {ssim_t/len(loader):.4f} | LPIPS: {lpips_t/len(loader):.4f}")

if __name__ == '__main__':
    main()
