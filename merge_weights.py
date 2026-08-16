import torch
import argparse
import os

def average_weights(ckpt1_path, ckpt2_path, out_path):
    print(f"Loading {ckpt1_path}...")
    ckpt1 = torch.load(ckpt1_path, map_location='cpu', weights_only=False)
    state_dict1 = ckpt1['model_state_dict']
    
    print(f"Loading {ckpt2_path}...")
    ckpt2 = torch.load(ckpt2_path, map_location='cpu', weights_only=False)
    state_dict2 = ckpt2['model_state_dict']
    
    print("Averaging weights...")
    averaged_state_dict = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2.0
        else:
            print(f"Warning: Key {key} not found in {ckpt2_path}!")
            averaged_state_dict[key] = state_dict1[key]
            
    # Check for keys in ckpt2 not in ckpt1
    for key in state_dict2.keys():
        if key not in state_dict1:
            print(f"Warning: Key {key} found in {ckpt2_path} but not in {ckpt1_path}!")
            averaged_state_dict[key] = state_dict2[key]
            
    # Create new checkpoint dict, inheriting config from ckpt2 (best model)
    new_ckpt = {
        'model_state_dict': averaged_state_dict,
        'config': ckpt2.get('config', {}),
        'scale': ckpt2.get('scale', 2)
    }
    
    print(f"Saving averaged checkpoint to {out_path}...")
    torch.save(new_ckpt, out_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', type=str, required=True)
    parser.add_argument('--ckpt2', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    
    average_weights(args.ckpt1, args.ckpt2, args.out)
