import torch

from .evaluator import Evaluator


class PSNREvaluator(Evaluator):
    @torch.inference_mode
    def eval(self, model) -> float:
        model.eval()
        model.requires_grad_(False)

        sse = 0
        pixels = 0
        for batch in self.dataloader:
            model_input = model.preprocessor.encode(batch[0].to(self.device))
            target = model.preprocessor.encode(batch[1].to(self.device))
    
            output = model.generator(model_input)
            
            sse += torch.sum((output - target) ** 2)
            pixels += output.numel()
        
        mse = sse / pixels

        psnr = 10 * torch.log10(1.0 / mse)
        
        return psnr

    @property
    def name(self):
        return "psnr"

