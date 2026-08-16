import torch
import torch.nn as nn
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

def heteroscedastic_loss(pred, gt, sigma):
    # L1-based heteroscedastic loss
    # sigma is expected to be strictly positive (e.g. from Softplus + epsilon)
    return torch.mean(torch.abs(pred - gt) / sigma + torch.log(sigma))

class RestorationLoss(nn.Module):
    def __init__(self, lpips_model, lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.5, lambda_hetero=0.0, device='cuda'):
        super(RestorationLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_hetero = lambda_hetero
        
        self.l1_loss = nn.L1Loss()
        # SSIM from torchmetrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # LPIPS expects 3 channels, we will repeat grayscale channels
        self.lpips_loss = lpips_model
        self.device = device

    def forward(self, pred, gt, sigma=None):
        # 1. L1 Loss (or Heteroscedastic Loss if sigma is provided)
        if sigma is not None and self.lambda_hetero > 0.0:
            loss_pixel = heteroscedastic_loss(pred, gt, sigma)
            pixel_weight = self.lambda_hetero
            l1_val = loss_pixel.item() # for logging purposes
        else:
            loss_pixel = self.l1_loss(pred, gt)
            pixel_weight = self.lambda_l1
            l1_val = loss_pixel.item()
        
        # 2. SSIM Loss (1 - SSIM)
        # ssim outputs a tensor scalar
        ssim_val = self.ssim(pred, gt)
        loss_ssim = 1.0 - ssim_val
        
        # 3. LPIPS Loss
        # LPIPS expects inputs in [-1, 1] and 3 channels
        pred_3c = pred.repeat(1, 3, 1, 1)
        gt_3c = gt.repeat(1, 3, 1, 1)
        
        # Convert [0, 1] to [-1, 1]
        pred_lpips_in = pred_3c * 2.0 - 1.0
        gt_lpips_in = gt_3c * 2.0 - 1.0
        
        loss_lpips = self.lpips_loss(pred_lpips_in, gt_lpips_in).mean()
        
        # Total Loss
        total_loss = (pixel_weight * loss_pixel + 
                      self.lambda_ssim * loss_ssim + 
                      self.lambda_lpips * loss_lpips)
                      
        return total_loss, l1_val, ssim_val.item(), loss_lpips.item()
