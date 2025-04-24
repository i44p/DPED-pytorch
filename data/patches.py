import torch
import numpy as np
import einops
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from pathlib import Path

class DPEDPatchDataset(Dataset):
    def __init__(self, path: Path, input_label, target_label, config):
        self.path = Path(path)
        self.input_label = input_label
        self.target_label = target_label
        self.config = config
        
        assert self.path.is_dir()

        self.len = len(list((self.path/input_label).iterdir()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_patch = np.asarray(Image.open(self.path / self.input_label / f"{idx}.jpg"))
        target_patch = np.asarray(Image.open(self.path / self.target_label / f"{idx}.jpg"))

        return (
            einops.rearrange(torch.from_numpy(input_patch.copy()), "b h w c -> b c h w"),
            einops.rearrange(torch.from_numpy(target_patch.copy()), "b h w c -> b c h w")
        )

    def get_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size = self.config.hyperparameters.batch_size,
            shuffle=True,
            num_workers=8,
            prefetch_factor=3,
            pin_memory=True,
        )
        return dataloader

