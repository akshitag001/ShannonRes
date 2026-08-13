import os
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
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .npy files")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[os.path.join("weights", "best_model.pth")], help="Paths to model checkpoints for ensembling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")
    
    # Load all models for ensembling
    models = []
    
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
            
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        scale = checkpoint['scale']
        
        model = RestorationCNN(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            num_features=config['num_features'],
            num_res_blocks=config['num_res_blocks'],
            scale=scale
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
    print(f"Successfully loaded {len(models)} model(s) for ensembling.")

    # DataLoader for batching
    dataset = InferenceDataset(args.input_dir)
    # Use batch_size from config or default to 32 for inference
    batch_size = config.get('batch_size', 32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def inference_with_tta(x, model):
        outputs = []
        for k in range(4):
            # Rotate
            rot_x = torch.rot90(x, k, dims=[2, 3])
            out_rot = model(rot_x)
            outputs.append(torch.rot90(out_rot, -k, dims=[2, 3]))
            
            # Rotate + Flip
            flip_x = torch.flip(rot_x, [3])
            out_flip = model(flip_x)
            outputs.append(torch.rot90(torch.flip(out_flip, [3]), -k, dims=[2, 3]))
            
        return torch.mean(torch.stack(outputs), dim=0)

    with torch.no_grad():
        for batch_tensors, filenames in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            # Inference with 8x TTA across all ensembled models
            ensemble_outputs = []
            for m in models:
                ensemble_outputs.append(inference_with_tta(batch_tensors, m))
                
            outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)
            
            # Post-process and save
            outputs = outputs.cpu().numpy()
            
            for i in range(len(filenames)):
                filename = filenames[i]
                # Squeeze channel dim (1, H, W) -> (H, W) to match input shape
                out_arr = outputs[i].squeeze(0)
                
                save_path = os.path.join(args.output_dir, filename)
                np.save(save_path, out_arr)
                
    print(f"Successfully processed {len(dataset)} images and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
