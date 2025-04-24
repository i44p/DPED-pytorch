import torch
import numpy as np
import h5py as h5
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from pathlib import Path

class DPEDPatchDataset(Dataset):
    def __init__(self, path: Path, input_label, target_label, config):
        self.path = Path(path)
        self.input_label = input_label
        self.target_label = target_label
        self.config = config
        
        assert self.path.is_file()

        with h5.File(self.path, 'r') as d:
            input_dataset = d[f"{self.input_label}/pixels"]
            self.len = input_dataset.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5.File(self.path, 'r') as d:
            input_patch = d[f"{self.input_label}/pixels"][idx]
            target_patch = d[f"{self.target_label}/pixels"][idx]

        return torch.from_numpy(input_patch.copy()), torch.from_numpy(target_patch.copy())
    
    def get_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size = self.config.hyperparameters.batch_size,
            shuffle=True,
            num_workers=8,
            prefetch_factor=3,
            pin_memory=True
        )
        return dataloader

