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

        self._params_to_optim, self.generator, self.discriminator = self._prepare_models()
        self.criterion = self._prepare_criterion()
        
    def _prepare_models(self):
        params_to_optim = []
        
        generator = import_class(self.config.model.generator.module)().to(self.device)
        params_to_optim.append(
            {
                "params": list(generator.parameters())
            }
        )
        generator.train(True)

        discriminator = import_class(self.config.model.discriminator.module)().to(self.device)
        params_to_optim.append(
            {
                "params": list(discriminator.parameters())
            }
        )

    
        if self.config.trainer.get('resume_path'):
            load_model(self, self.config.trainer.resume_path)

        return params_to_optim, generator, discriminator
    
    def _prepare_criterion(self):
        return import_class(self.config.criterion.get('module', torch.nn.MSELoss))(
                self,
                **self.config.criterion.get("args", {'reduction': 'none'})
            )
    
    def get_parameters_to_optimize(self):
        return self._params_to_optim
    
    def forward(self, model_input, target):

        output = self.generator(model_input)

        loss = self.criterion(output, target)
        return loss
