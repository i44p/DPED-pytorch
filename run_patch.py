import argparse

import torch
from safetensors.torch import load_model
from PIL import Image

from omegaconf import OmegaConf

import data.utils

from modules.dped import DPEDModel

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("config")
parser.add_argument("input_image")
parser.add_argument("output_image")

args = parser.parse_args()

config = OmegaConf.load(args.config)

model = DPEDModel(config, 'cpu')
load_model(model, args.model)

@torch.no_grad
def infer():
    img = data.utils.load_image(args.input_image, min_=0).unsqueeze(0)

    out_img = model.generator(img).squeeze()

    data.utils.save_image(out_img, args.output_image, min_=0)

infer()