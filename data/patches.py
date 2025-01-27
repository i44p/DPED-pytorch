import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from .utils import load_image


class DPEDPatchDataset(Dataset):
    def __init__(self, path: Path, input_label, target_label, config):
        self.path = Path(path)
        self.input_label = input_label
        self.target_label = target_label
        self.config = config
        
        norms = config.model.get("image_normalization", {"min": 0, "max": 1})
        self.min_, self.max_ = float(norms["min"]), float(norms["max"])

        assert self.path.is_dir()

        self.len = len(list((self.path/input_label).iterdir()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_patch = load_image(self.path / self.input_label / f"{idx}.jpg", self.min_, self.max_)
        target_patch = load_image(self.path / self.target_label / f"{idx}.jpg", self.min_, self.max_)
        
        return input_patch, target_patch

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

