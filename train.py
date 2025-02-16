
import argparse
import pathlib

from omegaconf import OmegaConf

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=pathlib.Path)
    args = parser.parse_args()

    return OmegaConf.load(args.config)

if __name__ == '__main__':
    config = get_config()

import os
import torch

torch.set_num_threads(os.cpu_count())

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer(config)
    trainer.train()

