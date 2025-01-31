from safetensors.torch import load_file
import torch
import sys

state_dict = load_file(sys.argv[1])

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")