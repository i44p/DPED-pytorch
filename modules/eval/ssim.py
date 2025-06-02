import torch

from .evaluator import Evaluator

from pytorch_msssim import ssim


class SSIMEvaluator(Evaluator):
    @torch.inference_mode
    def eval(self, model) -> float:
        model.eval()
        model.requires_grad_(False)

        ssim_accumulator = 0
        for batch in self.dataloader:
            model_input = model.preprocessor.encode(batch[0].to(self.device))
            target = model.preprocessor.encode(batch[1].to(self.device))
    
            output = model.generator(model_input)
            
            ssim_accumulator += torch.sum(ssim(output, target, data_range=1, size_average=False))
        
        metric = ssim_accumulator / self.dataloader.batch_size / len(self.dataloader)
        
        return metric
    
    def eval_batch(self, model, model_input, target) -> float:
        model.eval()
        model.requires_grad_(False)

        output = model.generator(model_input)

        metric = torch.mean(ssim(output, target, data_range=1, size_average=False))

        return metric

    @property
    def name(self):
        return "ssim"
