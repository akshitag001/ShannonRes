import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import RestorationCNN

class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.input_dir, filename)
        
        # Load and prepare shape (1, H, W)
        arr = np.load(filepath).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, filename

def main():
    parser = argparse.ArgumentParser(description="Inference for Image Restoration")
    parser.add_argument("input_dir", type=str, help="Directory containing input .npy files")
    parser.add_argument("output_dir", type=str, help="Directory to save output .npy files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")
    
    # Path to model checkpoint
    ckpt_path = os.path.join("weights", "best_model_seed2.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please place the weights properly.")
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    scale = checkpoint.get('scale', 2)
    
    model = RestorationCNN(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        num_features=config.get('num_features', 64),
        num_res_blocks=config.get('num_res_blocks', 16),
        scale=scale
    ).to(device)
    
    # Check if model has predicting uncertainty in its state dict but we don't want it for standard inference
    strict_load = True
    for key in checkpoint['model_state_dict'].keys():
        if "conv_uncertainty" in key:
            strict_load = False
            break
            
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict_load)
    model.eval()
    
    original_model = model
    try:
        compiled_model = torch.compile(model)
        # Dummy pass to trigger compilation and catch Triton errors immediately
        dummy_input = torch.randn(1, config.get('in_channels', 1), 64, 64).to(device)
        with torch.no_grad():
            compiled_model(dummy_input)
        model = compiled_model
        print("Successfully compiled model with torch.compile()")
    except Exception as e:
        print(f"Warning: torch.compile() failed, falling back to uncompiled model. Error: {e}")
        model = original_model

    dataset = InferenceDataset(args.input_dir)
    batch_size = config.get('batch_size', 16) # keep it reasonable
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch_tensors, filenames in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            outputs = model(batch_tensors)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            outputs = outputs.cpu().numpy()
            
            for i in range(len(filenames)):
                filename = filenames[i]
                # Squeeze channel dim (1, H, W) -> (H, W) to match input shape
                out_arr = outputs[i].squeeze(0)
                
                # Replace NaNs and Infs
                out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Clip values to strictly [0, 1]
                out_arr = np.clip(out_arr, 0.0, 1.0)
                
                save_path = os.path.join(args.output_dir, filename)
                np.save(save_path, out_arr)
                
    print(f"Successfully processed {len(dataset)} images and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
