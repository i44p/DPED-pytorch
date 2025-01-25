import argparse

import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from safetensors.torch import load_model
from PIL import Image

from modules import DPEDGenerator

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("input_image")
parser.add_argument("output_image")

args = parser.parse_args()

model = DPEDGenerator()
load_model(model, args.model)

img = pil_to_tensor(Image.open(args.input_image)).float() / 255 - 0.5

out_img = (model(img) + 0.5) * 255

out_img = to_pil_image(out_img).save(args.output_image)
