import argparse

import torch
from safetensors.torch import load_model
from PIL import Image

import data.utils

from modules import DPEDGenerator

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("input_image")
parser.add_argument("output_image")

args = parser.parse_args()

model = DPEDGenerator()
load_model(model, args.model)

img = data.utils.load_image(args.input_image).unsqueeze(0)

out_img = model(img).squeeze()

data.utils.save_image(out_img, args.output_image)
