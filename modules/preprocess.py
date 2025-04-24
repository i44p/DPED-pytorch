import torch
import numpy as np
from PIL import Image


class DPEDProcessor(torch.nn.Module):
    def __init__(
        self,
        norm_min,
        norm_max,
    ):
        self.min = float(norm_min)
        self.max = float(norm_max)

        self._norm_mult = (self.max - self.min) / 255.0
        self._denorm_mult = 255 / (self.max - self.min)
    
    def from_pil(self, image: Image.Image) -> torch.Tensor:
        return self.from_numpy(np.asarray(image).copy())
    
    def pil(self, batch: torch.Tensor) -> Image.Image:
        return Image.fromarray(self.numpy(batch))
    
    def from_numpy(self, batch: np.ndarray) -> torch.Tensor:
        return self.encode(torch.from_numpy(batch))
    
    def numpy(self, batch: torch.Tensor) -> np.ndarray:
        decoded = self.decode(batch)
        return decoded.detach().round().to(torch.uint8).cpu().numpy()
    
    @staticmethod
    def permute_pil_to_tensor(batch: torch.Tensor) -> torch.Tensor:
        return batch.permute(0, 3, 1, 2)

    @staticmethod
    def permute_tensor_to_pil(batch: torch.Tensor) -> torch.Tensor:
        return batch.permute(0, 2, 3, 1)
    
    def encode(self, batch):
        batch = self.permute_pil_to_tensor(batch)
        return self.normalize_batch(batch)
    
    def decode(self, batch):
        batch = self.permute_tensor_to_pil(batch)
        return self.denormalize_batch(batch)
    
    def normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.float() * self._norm_mult + self.min

    def denormalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch.float() - self.min) * self._denorm_mult


