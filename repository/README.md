# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview
This repository contains the solution for Phase 1 of the SEMICON India Hackathon 2026 (KLA problem statement). The solution implements a Single-stage Convolutional Neural Network (SRResNet-style) to perform blind, compound-degradation restoration and super-resolution on noisy grayscale semiconductor images.

### Degradations Addressed:
- Speckle Noise (multiplicative)
- Additive Gaussian Noise
- Downsampling (dynamically inferred scale factor, typically 2x)

### Assumptions:
- **Grayscale Images**: Both ground truth (GT) and noisy (NoisyLR) images are 1-channel grayscale arrays.
- **Data Format**: Images are provided as `.npy` arrays.
- **Scale Factor**: The model infers the scale factor dynamically by comparing the GT and NoisyLR spatial dimensions at runtime.
- **Value Ranges**: Ground truth arrays are strictly in `[0, 1]`. NoisyLR arrays may exceed `[0, 1]` due to noise properties. The model clamps predictions to `[0, 1]` at the output layer.

## Environment Setup
It is recommended to run this project in a clean Python 3.10+ environment with an NVIDIA GPU.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure
- `train.py`: Reproducible training script with fixed seeds and hyperparameter logging.
- `inference.py`: Standalone, optimized batch inference script that strictly follows the requested CLI signature.
- `configs/default.yaml`: Hyperparameters for training.
- `src/`: Core logic (model architecture, data loading, composite loss, metrics).

## Training the Model
To train the model from scratch on the provided dataset:
```bash
python train.py --config configs/default.yaml
```
Metrics (PSNR, SSIM, LPIPS) will be evaluated on a 10% validation split carved from the training data. Checkpoints are automatically saved to `weights/best_model.pth`.

## Running Inference
The inference script expects a directory of `.npy` arrays and outputs restored arrays of the exact same filename and `.npy` format.

```bash
python inference.py --input_dir path/to/Test_NoisyLR/NoisyLR --output_dir path/to/output_restored
```
*Note: The script loads the trained weights from `weights/best_model.pth`. Ensure you have trained the model or placed the bundled checkpoint there before running inference.*
