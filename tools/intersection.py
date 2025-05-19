import kornia
import numpy as np
import torch

from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from einops import rearrange


class Intersection:
    def __init__(self, threshold=0.2):
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
        self.model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")
        self.threshold = threshold
    
    @torch.inference_mode()
    def get_keypoints(self, input: Image.Image, target: Image.Image):
        images = [input, target]

        inputs = self.processor(images, return_tensors="pt")
        outputs = self.model(**inputs)

        image_sizes = [[(image.height, image.width) for image in images]]
        outputs = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.threshold)
        return outputs

    @torch.inference_mode()
    def get_homography(self, keypoints1, keypoints2):
        return kornia.geometry.homography.find_homography_dlt(keypoints1, keypoints2)
    
    @torch.inference_mode()
    def warp_image(self, base: torch.Tensor, H, dsize) -> torch.Tensor:
        return kornia.geometry.transform.warp_perspective(base, H, dsize=dsize, align_corners=True)
    
    @torch.inference_mode()
    def intersect_single(self, input_data: Image.Image, target_data: Image.Image):
        kp = self.get_keypoints(input_data, target_data)[0]
        keypoints_input, keypoints_target = kp['keypoints0'].float(), kp['keypoints1'].float()

        H = self.get_homography(keypoints_input.unsqueeze(0), keypoints_target.unsqueeze(0))

        input_torch = rearrange(
            torch.from_numpy(np.asarray(input_data).copy()).float() / 255,
            'h w c -> c h w'
        )
        warped_input = self.warp_image(
            input_torch.unsqueeze(0),
            H, dsize=(target_data.height, target_data.width)
        )

        return H, warped_input
