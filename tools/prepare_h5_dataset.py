import argparse
import pathlib

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dataset', type=pathlib.Path,
                        help="Path to the directory filled with {%%06d}.{ext}-named 'fake' images (ie kvadra)")
    parser.add_argument('target_dataset', type=pathlib.Path,
                        help="Path to the directory filled with {%%06d}.{ext}-named 'real' images (ie sony)")

    parser.add_argument('output_path', type=pathlib.Path,
                        help="Path to save h5 dataset.")
    
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to run intersection model on.")

    parser.add_argument('-a', '--append', action='store_true',
                        help="Push the photos to the back of the dataset instead of creating a new one.")
    
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help="")
    parser.add_argument('-s', '--seed', type=int, default=1337,
                        help="not implemented")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="")
    parser.add_argument('--prefetch_factor', type=int, default=3,
                        help="")
    parser.add_argument('--shuffle', type=bool, default=False,
                        help="")
    parser.add_argument('--raw', type=bool, default=False,
                        help="Enable processing of raw files (not implemented)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

from PIL import Image

import os
import torch
torch.set_num_threads(os.cpu_count())
import safetensors
import h5py
import pathlib
import rawpy
import numpy as np
import io
import traceback

from tqdm import tqdm
from einops import rearrange

import tools.intersection as intersection


def is_simple_image_file(fn: pathlib.Path):
    exts = ['.jpg', '.png', '.jpeg']
    
    return fn.suffix.lower() in exts


class CommonDataset(torch.utils.data.Dataset):
    def __init__(self, input_path: pathlib.Path, target_path: pathlib.Path):
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path

        assert self.input_path.is_dir()
        assert self.target_path.is_dir()

        self.file_list = self._get_file_list()
        self._len = len(self.file_list)

    def __len__(self):
        return self._len
    
    def _get_file_list(self):
        file_list = [fn.name for fn in self.target_path.iterdir() if is_simple_image_file(fn)]
        file_list.sort(key=lambda fn: int(fn.split('.')[0]))
        return file_list

    def __getitem__(self, idx):        
        input_fp = self.fetch_filename(self.input_path, idx)
        target_fp = self.fetch_filename(self.target_path, idx)

        with Image.open(input_fp) as img:
            input_tensor = pil2torch(img)
        
        with Image.open(target_fp) as img:
            target_tensor = pil2torch(img)
        
        return input_tensor, target_tensor
    
    def fetch_filename(self, path: pathlib.Path, idx):
        # pattern = f'{idx:06d}.*'
        # input_guesses = list(path.glob(pattern))
        # file_path = input_guesses[0]
        file_path = path / self.file_list[idx]
        return file_path


def pil2torch(img: Image.Image) -> torch.Tensor:
    return rearrange(torch.from_numpy(np.asarray(img).copy()).float() / 255, 'h w c -> c h w')

def torch2pil(img: torch.Tensor) -> Image.Image:
    return Image.fromarray(rearrange((img * 255).numpy().astype('uint8'), 'c h w -> h w c'))

def extend_dataset(dataset, batch):
    if np.all(dataset[0] == 0):
        dataset.resize(batch.shape[0], axis=0)
    else:
        dataset.resize(dataset.shape[0] + batch.shape[0], axis=0)

    dataset[-batch.shape[0]:] = batch

def get_datasets(dataset, name, shape, dtype='uint8'):
    batch, h, w, c = shape
    new_dataset = dataset.require_dataset(
        name,
        shape=shape,
        maxshape=(None, h, w, c),
        compression="lzf",
        chunks=(1, h, w, c),
        #compression_opts=5,
        dtype=dtype
        )

    return new_dataset


def main(args):
    dataset = CommonDataset(args.input_dataset, args.target_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
    )

    c, h, w = dataset[0][0].shape

    slicer = intersection.Intersection(device=args.device)

    open_mode = "a" if args.append else "w"
    with h5py.File(args.output_path, open_mode, libver='latest') as dataset:

        input_dataset = get_datasets(dataset, 'input', (args.batch_size, h, w, c))
        target_dataset = get_datasets(dataset, 'target', (args.batch_size, h, w, c))

        step_bar = tqdm(dataloader, desc='batch')
        for batch_idx, batch in enumerate(step_bar):
            input_batch = batch[0]
            target_batch = batch[1]

            #Hs, warped_target_batch = slicer.intersect(input_batch, target_batch)
            warped_target_batch = []
            masked_input_batch = []
            for idx, (inp, tgt) in enumerate(zip(input_batch, target_batch)):
                try:
                    H, warped_target = slicer.intersect_single(torch2pil(inp), torch2pil(tgt))
                except Exception as e:
                    print("INDEX:", batch_idx)
                    print("SUBINDEX:", idx)
                    print("Failed to find keypoints/homography. Skipping.")
                    print(traceback.format_exc())
                    continue
                mask = torch.where(warped_target == 0)
                masked_input = inp
                masked_input[mask] = 0
                warped_target_batch.append(warped_target)
                masked_input_batch.append(masked_input)
            warped_target_batch = torch.stack(warped_target_batch).cpu()
            masked_input_batch = torch.stack(masked_input_batch)

            # kvadra: rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace(5))
            # apparently kvadra's camera app writes DNGs in XYZ colorspace

            extend_dataset(input_dataset, rearrange(masked_input_batch.numpy(), 'b c h w -> b h w c') * 255)
            extend_dataset(target_dataset, rearrange(warped_target_batch.numpy(), 'b c h w -> b h w c') * 255)


if __name__ == '__main__':
    main(args)