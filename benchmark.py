import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import ImageRestorationDataset
from src.model import RestorationCNN

def generate_visuals(device, model_path="weights/best_model.pth", out_path="visual_results.png"):
    print(f"\n--- Generating Visual Results ---")
    if not os.path.exists(model_path):
        print(f"Skipping visualization: Model not found at {model_path}. Wait for training to save it.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = RestorationCNN(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_features=config['num_features'],
        num_res_blocks=config['num_res_blocks'],
        scale=checkpoint['scale']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_dataset = ImageRestorationDataset(
        gt_dir=config['train_gt_dir'],
        noisy_dir=config['train_noisy_dir'],
        gt_crop_size=config.get('gt_crop_size', 128),
        synthesize_prob=0.0
    )
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    with torch.no_grad():
        for i in range(3):
            # Grab sequential or random samples
            idx = i * (len(val_dataset) // 3)
            noisy_tensor, gt_tensor, filename, is_synthetic = val_dataset[idx]
            
            # Add batch dimension
            noisy_input = noisy_tensor.unsqueeze(0).to(device)
            
            # Predict
            restored_output = model(noisy_input).squeeze(0).cpu()
            
            # Visualize
            noisy_np = noisy_tensor.squeeze(0).numpy()
            restored_np = restored_output.squeeze(0).numpy()
            gt_np = gt_tensor.squeeze(0).numpy()
            
            axes[i, 0].imshow(noisy_np, cmap='gray')
            axes[i, 0].set_title(f"Input NoisyLR\n(Shape: {noisy_np.shape})")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(restored_np, cmap='gray')
            axes[i, 1].set_title(f"Restored Output\n(Shape: {restored_np.shape})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(gt_np, cmap='gray')
            axes[i, 2].set_title(f"Ground Truth\n(Shape: {gt_np.shape})")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved visual grid to {out_path}")

def run_benchmark(device, model_path="weights/best_model.pth"):
    print(f"\n--- Running Inference Latency Benchmark ---")
    if not os.path.exists(model_path):
        print(f"Skipping benchmark: Model not found at {model_path}. Wait for training to save it.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = RestorationCNN(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_features=config['num_features'],
        num_res_blocks=config['num_res_blocks'],
        scale=checkpoint['scale']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dummy tensor matching expected input size (1, 128, 128)
    dummy_input = torch.rand(1, 1, 128, 128).to(device)

    # Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # Benchmark
    num_runs = 500
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    # For CUDA, we should ideally synchronize before ending the timer.
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_image_ms = (total_time / num_runs) * 1000
    throughput = num_runs / total_time
    
    print(f"Device: {device.type.upper()}")
    print(f"Total time for {num_runs} inferences: {total_time:.3f} seconds")
    print(f"Average Inference Time per image: {time_per_image_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/second")

if __name__ == "__main__":
    import yaml
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    benchmark_file = os.path.join("weights", "best_model.pth")
    if os.path.exists(benchmark_file):
        generate_visuals(device, benchmark_file)
        run_benchmark(device, benchmark_file)
    else:
        print(f"Waiting for best_model.pth to be saved to weights/ ... Run this script after training.")
