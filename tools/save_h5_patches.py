import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from einops import rearrange
import numpy as np

import argparse
import tqdm

from data.h5 import H5Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-p', '--patch_size', type=int)
    parser.add_argument('-pd', '--padding_px', type=int)
    parser.add_argument('-gl', '--guess_limit', type=int)
    parser.add_argument('-w', '--workers', type=int)
    parser.add_argument('-ppi', '--num_patches_per_image', type=int)
    parser.add_argument('-c', '--correlation_threshold', type=float)
    parser.add_argument('--save_pairs', action='store_true')
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('output_path', type=Path)
    args = parser.parse_args()

    dataset = H5Dataset(
        args.dataset_path,
        **{k:v for k,v in vars(args).items() if v is not None}
    )

    print(f"Extracting patches from {len(dataset)} images.")

    dataloader = dataset.get_dataloader()

    n = 0

    input_save_path = args.output_path / 'input'
    input_save_path.mkdir(exist_ok=True, parents=True)

    target_save_path = args.output_path / 'target'
    target_save_path.mkdir(exist_ok=True, parents=True)

    if args.save_pairs:
        pair_save_path = args.output_path / 'pairs'
        pair_save_path.mkdir(exist_ok=True, parents=True)

    for batch in tqdm.tqdm(dataloader):
        batch_input = rearrange(batch[0], 'b c h w -> b h w c').numpy()
        batch_target = rearrange(batch[1], 'b c h w -> b h w c').numpy()
        
        for in_patch, tgt_patch in zip(batch_input, batch_target):
            in_img = Image.fromarray(in_patch)
            tgt_img = Image.fromarray(tgt_patch)

            save_name = f"{n}.png"

            in_img.save(input_save_path / save_name)
            tgt_img.save(target_save_path / save_name)

            if args.save_pairs:
                concat_img = Image.fromarray(np.concatenate((in_patch, tgt_patch), axis=1))
                concat_img.save(pair_save_path / save_name)

            n += 1

