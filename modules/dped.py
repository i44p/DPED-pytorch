import torch
import torch.nn as nn

from .discriminator import DPEDDiscriminator
from .generator import DPEDGenerator

from class_utils import import_class

class DPEDModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self.generator = self._prepare_models()
        self.criterion = self._prepare_criterion()
        
    def _prepare_models(self):
        generator = import_class(self.config.model.generator.module)().to(self.device)
        trainer = self.config.trainer
        if trainer.get('resume_path'):
            load_model(generator, traner.resume_path)
        generator.train(True)
        return generator
    
    def _prepare_criterion(self):
        self.criterion = import_class(self.config.criterion.get('module', torch.nn.MSELoss))(
                self,
                **self.config.criterion.get("args", {'reduction': 'none'})
            )
    
    def get_parameters_to_optimize(self):
        params_to_optim = []

        for model in [self.generator]:
            params_to_optim.append(
                {
                    "params": list(model.parameters())
                }
            )
            model.train(True)
        
        return params_to_optim
    
    def forward(self, model_input, target):

        output = self.generator(model_input)

        loss = self.criterion(output, target)
        return loss
