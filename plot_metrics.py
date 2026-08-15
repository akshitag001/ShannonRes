import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path, out_prefix):
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    
    # 1. Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['train_loss'], color='red', linewidth=2)
    plt.title('Training Loss Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_loss.png', dpi=300)
    plt.close()

    # 2. Validation PSNR
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['val_psnr'], color='blue', linewidth=2)
    plt.title('Validation PSNR over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_psnr.png', dpi=300)
    plt.close()

    # 3. Validation SSIM
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['val_ssim'], color='green', linewidth=2)
    plt.title('Validation SSIM over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_ssim.png', dpi=300)
    plt.close()

    # 4. Validation LPIPS
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['val_lpips'], color='purple', linewidth=2)
    plt.title('Validation LPIPS over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('LPIPS (Lower is better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_lpips.png', dpi=300)
    plt.close()

csv_file = r"C:\Users\24bcscs005\.gemini\antigravity-ide\brain\dc45a59e-441f-40fa-bfa4-59ff5b11abac\train_log_seed2.csv"
out_base = r"C:\Users\24bcscs005\.gemini\antigravity-ide\brain\dc45a59e-441f-40fa-bfa4-59ff5b11abac\convergence_seed2"

plot_metrics(csv_file, out_base)
print("Plots generated successfully.")
