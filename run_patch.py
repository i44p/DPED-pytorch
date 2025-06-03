import argparse
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("config")
parser.add_argument("input_path", type=pathlib.Path)
parser.add_argument("output_path", type=pathlib.Path)

args = parser.parse_args()

from gradio_app import DPED
import torch
from PIL import Image

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']


@torch.inference_mode()
def main(model, input_image, output_image):
    with Image.open(input_image) as i:
        model.infer(i).save(output_image)

if __name__ == '__main__':
    model = DPED(args.config, args.model)
    if args.input_path.is_file() and (not args.output_path.exists() or args.output_path.is_file()):
        main(model, args.input_path, args.output_path)
    
    elif args.input_path.is_dir() and args.output_path.is_dir():
        for file in args.input_path.glob('*'):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                main(model, str(file), str(args.output_path / (file.stem + '.png')))
    else:
        print("Couldn't parse input or output path.")
