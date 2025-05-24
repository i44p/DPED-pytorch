import torch

from .evaluator import Evaluator

from pytorch_msssim import ms_ssim


class MSSSIMEvaluator(Evaluator):
    @torch.inference_mode
    def eval(self, model) -> float:
        model.eval()
        model.requires_grad_(False)

        ms_ssim_accumulator = 0
        for batch in self.dataloader:
            model_input = model.preprocessor.encode(batch[0].to(self.device))
            target = model.preprocessor.encode(batch[1].to(self.device))
    
            output = model.generator(model_input)
            
            ms_ssim_accumulator += torch.sum(ms_ssim(
                output, target,
                win_sigma=1.5,
                win_size=11,
                K=(0.01, 0.03),
                data_range=1,
                size_average=False
                )
            )
        
        metric = ms_ssim_accumulator / self.dataloader.batch_size / len(self.dataloader)
        
        return metric

    @property
    def name(self):
        return "ms-ssim"
