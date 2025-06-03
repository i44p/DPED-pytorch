import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("config")
parser.add_argument("input_image")
parser.add_argument("output_image")

args = parser.parse_args()

from gradio_app import DPED
import torch
from PIL import Image


@torch.inference_mode()
def main(model, input_image, output_image):
    with Image.open(input_image) as i:
        model.infer(i).save(output_image)

if __name__ == '__main__':
    model = DPED(args.config, args.model)
    main(model, args.input_image, args.output_image)