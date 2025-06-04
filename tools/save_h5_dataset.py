import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import argparse
import h5py as h5
import tqdm
from time import perf_counter

class H5Dataset(Dataset):
    def __init__(self, path: Path, save_to: Path):
        self.path = Path(path)
        self.save_to = Path(save_to)
        
        assert self.path.is_file()

        with h5.File(self.path, 'r') as d:
            input_dataset = d[f"input"]
            self._len = input_dataset.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start = perf_counter()
        with h5.File(self.path, 'r') as d:
            input_img = d[f"input"][idx]
            target_img = d[f"target"][idx]
        elapsed = perf_counter() - start
        

        input_pil = Image.fromarray(input_img.copy())
        target_pil = Image.fromarray(target_img.copy())

        input_pil.save(self.save_to / 'input' / f'{idx}.png')
        target_pil.save(self.save_to / 'target' / f'{idx}.png')
        
        return elapsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('output_path', type=Path)
    args = parser.parse_args()

    dataset = H5Dataset(args.dataset_path, save_to=args.output_path)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=6,
    )
    times = 0
    for elapsed in tqdm.tqdm(dataloader):
        times += elapsed.sum()
    print(f"average time opening a pair: {times / len(dataset)}")
