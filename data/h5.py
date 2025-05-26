import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py as h5
import scipy
import einops

from PIL import Image

from pathlib import Path

class H5PatchDataset(Dataset):
    def __init__(self, path: Path, input_label, target_label, batch_size):
        self.path = Path(path)
        self.input_label = input_label
        self.target_label = target_label
        self.batch_size = batch_size
        
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
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=8,
            prefetch_factor=3,
            pin_memory=True
        )
        return dataloader


class H5Dataset(Dataset):
    def __init__(self, path: Path,
            batch_size,
            patch_size = 100,
            correlation_threshold = 0.9,
            padding_px=10,
            guess_limit=2000,
            workers=4,
            *args, **kwargs
        ):
        self.path = Path(path)
        self.patch_size = patch_size
        self.workers = workers
        self.correlation_threshold = correlation_threshold
        self.padding_px = padding_px
        self.guess_limit = guess_limit
        self.pad = self.patch_size // 2 + self.padding_px
        self.batch_size = batch_size

        self.h5_file = None
        self.input_dataset = None
        self.target_dataset = None
        self._init_dataset()
        
        assert self.path.is_file()

        self._len = self.input_dataset.shape[0]

    def __len__(self):
        return self._len
            
    def _init_dataset(self):
        if self.h5_file is None:
            self.h5_file = h5.File(self.path, 'r')
            self.input_dataset = self.h5_file["input"]
            self.target_dataset = self.h5_file["target"]
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
    
    def _find_patch_coords(self, input_img, target_img):
        x_center = None
        y_center = None

        h, w = input_img.shape
        corel = 0
        attempts = 0
        while attempts <= self.guess_limit and corel < self.correlation_threshold:
            attempts += 1
            
            x_center = np.random.randint(self.pad, h - self.pad)
            y_center = np.random.randint(self.pad, w - self.pad)

            input_patch = self._crop(input_img, y_center, x_center)
            target_patch = self._crop(target_img, y_center, x_center)

            if np.std(input_patch) == 0 or np.std(target_patch) == 0:
                continue

            corel_statistic, _ = scipy.stats.pearsonr(
                input_patch,
                target_patch,
                axis=None
            )

            if np.isnan(corel_statistic):
                continue
            
            corel = corel_statistic
        
        if attempts >= self.guess_limit:
            return None

        return (x_center, y_center)

    def __getitem__(self, idx):
        input_img = self.input_dataset[idx]
        target_img = self.target_dataset[idx]
        
        if np.all(input_img == 0):
            return None

        x_center, y_center = self._find_patch_coords(self._rgb2gray(input_img.astype(float) / 255), target_img)
        
        input_patch = self._crop(input_img, y_center, x_center)
        target_patch = self._crop(target_img, y_center, x_center)

        return (
            einops.rearrange(torch.from_numpy(input_patch.copy()), "h w c -> c h w"),
            einops.rearrange(torch.from_numpy(target_patch.copy()), "h w c -> c h w")
        )
    
    def _crop(self, img, y_center, x_center):
        return img[
            y_center - self.patch_size // 2 : y_center + self.patch_size // 2,
            x_center - self.patch_size // 2 : x_center + self.patch_size // 2,
            :,
        ]

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb, [299/1000, 587/1000, 114/1000])
    
    def get_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            prefetch_factor=3,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        return dataloader
    
    @staticmethod
    def collate_fn(batch):
        filtered_batch = [sample for sample in batch if sample is not None]
        
        # all samples in the batch are None
        if not filtered_batch:
            raise RuntimeError(f"\nGot empty batch, can not proceed.\n")

        if len(batch) != len(filtered_batch):
            print(f"\nSome items in the batch are None. Effective batch size: {len(filtered_batch)}\n")
        
        inp, target = zip(*filtered_batch)
        
        return (torch.stack(inp), torch.stack(target))

