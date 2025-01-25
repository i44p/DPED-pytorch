import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor

from pathlib import Path

from PIL import Image


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
        input_img = pil_to_tensor(Image.open(self.path / self.input_label / f"{idx}.jpg")).float() / 255 - 0.5
        target_img = pil_to_tensor(Image.open(self.path / self.target_label / f"{idx}.jpg")).float() / 255 - 0.5
        
        return input_img, target_img

    def get_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size = self.config.hyperparameters.batch_size,
            shuffle=True
        )
        return dataloader

