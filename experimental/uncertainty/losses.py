import torch

def heteroscedastic_loss(pred, gt, sigma):
    # L1-based heteroscedastic loss
    # sigma is expected to be strictly positive (e.g. from Softplus + epsilon)
    return torch.mean(torch.abs(pred - gt) / sigma + torch.log(sigma))
