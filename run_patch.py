import argparse

import torch
from safetensors.torch import load_model
from PIL import Image

from omegaconf import OmegaConf

from modules.dped import DPEDModel
from modules.preprocess import DPEDProcessor

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("config")
parser.add_argument("input_image")
parser.add_argument("output_image")

args = parser.parse_args()

config = OmegaConf.load(args.config)

processor = DPEDProcessor(**config.model.preprocessor.args)

model = DPEDModel(config, 'cpu')
load_model(model, args.model)


@torch.inference_mode()
def infer():
    img = processor.from_pil(Image.open(args.input_image))

    out_img = model.generator(img)

    processor.pil(out_img).save(args.output_image)

if __name__ == '__main__':
    infer()