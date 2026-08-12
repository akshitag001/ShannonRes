import torch
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class MetricsEvaluator:
    def __init__(self, lpips_model, device='cuda'):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_fn = lpips_model
        
    def evaluate(self, pred, gt):
        """
        Evaluate PSNR, SSIM, LPIPS for a batch.
        pred and gt should be in [0, 1] and shape (N, 1, H, W).
        """
        psnr_val = self.psnr(pred, gt).item()
        ssim_val = self.ssim(pred, gt).item()
        
        pred_3c = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
        gt_3c = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0
        
        # lpips returns (N, 1, 1, 1)
        lpips_val = self.lpips_fn(pred_3c, gt_3c).mean().item()
        
        return psnr_val, ssim_val, lpips_val
