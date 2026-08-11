import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + identity

class RestorationCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=16, scale=2):
        super(RestorationCNN, self).__init__()
        self.scale = scale
        
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(*res_blocks)
        
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Upsampling
        self.conv_up = nn.Conv2d(num_features, out_channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        
        # Final conv (optional, but good for refinement after PixelShuffle)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Feature extraction
        out_in = self.conv_in(x)
        
        # Body
        out = self.body(out_in)
        out = self.conv_body(out)
        out = out + out_in  # Global skip connection
        
        # Upsample
        out = self.conv_up(out)
        out = self.pixel_shuffle(out)
        
        # Final mapping
        out = self.conv_out(out)
        
        # Ensure output is in [0, 1] range (especially for inference/metrics)
        # Since it's continuous space, clamping might hurt gradients if it dies,
        # but torch.clamp handles gradients fine for the pass-through regions.
        out = torch.clamp(out, 0.0, 1.0)
        
        return out
