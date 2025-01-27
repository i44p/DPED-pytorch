import torch
import numpy as np

from pathlib import Path

from PIL import Image


def load_image(path: Path, min_=-1, max_=1):
    return normalize_image(pil_to_tensor(Image.open(path)), min_, max_)


def save_image(image: torch.Tensor, path: Path, min_=-1, max_=1):
    return tensor_to_pil(denormalize_image(image, min_, max_)).save(path)


def normalize_image(image: torch.Tensor, min_=-1, max_=1) -> torch.Tensor:
    return image.float() * (max_ - min_) / 255.0 + min_


def denormalize_image(image: torch.Tensor, min_=-1, max_=1) -> torch.Tensor:
    return (image.float() - min_) * 255 / (max_ - min_)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(tensor.detach().permute(1, 2, 0).numpy().round().astype(np.uint8))
