import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("config")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("-p", "--patch_size", default=256, type=int)
parser.add_argument("-s" ,"--stride", default=64, type=int)
parser.add_argument("-b" ,"--batch_size", default=64, type=int)

args = parser.parse_args()

from gradio_app import DPED
import torch
from PIL import Image
from itertools import batched
from tqdm import tqdm


@torch.inference_mode()
def main(args, model, input_image, output_image, patch_size, stride, batch_size):
    with Image.open(input_image) as i:
        pad_h = (patch_size - i.height % patch_size) % patch_size
        pad_w = (patch_size - i.width % patch_size) % patch_size
        img_torch = model.processor.from_pil(i)
        img_torch = torch.nn.functional.pad(img_torch, (0, pad_w, 0, pad_h))
    
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    fold = torch.nn.Fold(
            output_size=img_torch.shape[2:],
            kernel_size=patch_size,
            stride=stride,
        )
    
    # splitting the image
    patches = unfold(img_torch)
    L = patches.shape[2]
    patches = patches.permute(0, 2, 1).view(-1, img_torch.shape[1], patch_size, patch_size)

    # 2D hann window for blending
    window_1d = torch.hann_window(patch_size, device=img_torch.device, dtype=img_torch.dtype)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    window_2d = window_2d.view(1, 1, patch_size, patch_size)
    window_flat = window_2d.view(1, -1, 1)

    # divisor tensor for normalization (accumulates weights per pixel)
    ones = torch.ones(1, 1, *img_torch.shape[2:], device=img_torch.device, dtype=img_torch.dtype)
    ones_patches = unfold(ones)
    weighted_ones = ones_patches * window_flat
    divisor = fold(weighted_ones)
    divisor = divisor.clamp(min=1e-8)

    # batch processing
    output_patches = []
    batches = list(batched(patches, batch_size))
    for batch in tqdm(batches):
        batch = torch.stack(batch)
        out_batch = model.infer(batch).cpu()
        output_patches.append(out_batch)
    

    # assembling the image
    output_patches = torch.cat(output_patches, dim=0) * window_2d
    output_patches = output_patches.view(1, L, -1).permute(0, 2, 1)
    output = fold(output_patches) / 255.0
    divisor = divisor.expand_as(output)
    output = output / divisor
    output = output[:, :, :i.height, :i.width]

    model.processor.pil(output).save(output_image)
    

if __name__ == '__main__':
    model = DPED(args.config, args.model)
    main(args, model, args.input_image, args.output_image, args.patch_size, args.stride, args.batch_size)