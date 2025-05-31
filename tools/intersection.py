import kornia
import numpy as np
import torch

from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from einops import rearrange


class Intersection:
    def __init__(self, threshold=0.9, device='cpu'):
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
        self.model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor").to(device)
        self.threshold = threshold
        self.device = device
    
    @torch.inference_mode()
    def get_keypoints(
        self,
        inputs: list[Image.Image] | torch.Tensor,
        targets: list[Image.Image] | torch.Tensor,
        ):
        assert len(inputs) == len(targets), "Inputs and targets must be of the same length"

        images = []
        for inp, tgt in zip(inputs, targets):
            images.append([inp, tgt])
        
        processor_inputs = self.processor(
                images,
                return_tensors="pt",
                do_rescale=not isinstance(inputs, torch.Tensor)
            ).to(self.device)
        outputs = self.model(**processor_inputs)


        if isinstance(inputs[0], torch.Tensor):
            image_sizes = [[inp.shape[2:], tgt.shape[2:]] 
                    for inp, tgt in zip(inputs, targets)]
        else:
            image_sizes = [[(inp.height, inp.width), (tgt.height, tgt.width)] 
                    for inp, tgt in zip(inputs, targets)]
        
        processed_outputs = self.processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=self.threshold
        )
        return processed_outputs

    @torch.inference_mode()
    def get_homography(self, keypoints1, keypoints2):
        return kornia.geometry.homography.find_homography_dlt(keypoints1, keypoints2)
    
    @torch.inference_mode()
    def warp_image(self, base: torch.Tensor, H, dsize) -> torch.Tensor:
        return kornia.geometry.transform.warp_perspective(base, H, dsize=dsize, align_corners=True)
    
    @torch.inference_mode()
    def intersect_single(self, input_data: Image.Image, target_data: Image.Image):
        kps = self.get_keypoints((input_data,), (target_data,))

        keypoints_input, keypoints_target = [], []
        for kp in kps:
            keypoints_input.append(kp['keypoints0'].float())
            keypoints_target.append(kp['keypoints1'].float())

        keypoints_input = torch.stack(keypoints_input)
        keypoints_target = torch.stack(keypoints_target)

        Hs = self.get_homography(keypoints_target, keypoints_input)

        target_torch = rearrange(
            torch.from_numpy(np.asarray(target_data).copy()).float() / 255,
            'h w c -> c h w'
        ).unsqueeze(0).to(self.device)
        
        warped_target = self.warp_image(
            target_torch,
            Hs, dsize=(input_data.height, input_data.width)
        )

        return Hs[0], warped_target[0]
    
    @torch.inference_mode()
    def intersect(
        self,
        input_data: list[Image.Image] | torch.Tensor,
        target_data: list[Image.Image] | torch.Tensor,
        ):
        raise NotImplementedError
        
        if isinstance(input_data, torch.Tensor):
            kps = self.get_keypoints(input_data, target_data)
            
            _, _, height, width = input_data.shape
            input_torch = input_data
            target_torch = target_data
        else:
            kps = self.get_keypoints(input_data, target_data)
            height, width = input_data[0].height, input_data[0].width

            input_torch = torch.stack([
                rearrange(torch.from_numpy(np.asarray(data).copy()).float(), 'h w c -> c h w') / 255
                    for data in input_data
                ]
            )

            target_torch = torch.stack([
                    rearrange(torch.from_numpy(np.asarray(data).copy()).float(), 'h w c -> c h w') / 255
                    for data in target_data
                ]
            )

        # pad keypoints to align items inside the batch
        max_kp_len = max(kps, key=lambda kp: kp['keypoints0'].shape[0])['keypoints0'].shape[0]
        keypoints_input, keypoints_target = [], []
        for kp in kps:
            pad_amount = max_kp_len - kp['keypoints0'].shape[0]
            keypoints_input.append(self._pad_keypoints(pad_amount, kp['keypoints0'].float()))
            keypoints_target.append(self._pad_keypoints(pad_amount, kp['keypoints1'].float()))

        keypoints_input = torch.stack(keypoints_input)
        keypoints_target = torch.stack(keypoints_target)

        Hs = self.get_homography(keypoints_target, keypoints_input)
        
        warped_targets = self.warp_image(
            target_torch,
            Hs, dsize=(height, width)
        )

        return Hs, warped_targets
    
    @staticmethod
    def _pad_keypoints(pad_amount, keypoints, mode='constant', value=0):
        return torch.nn.functional.pad(keypoints, pad=[0, 0, pad_amount, 2], mode=mode, value=value)
