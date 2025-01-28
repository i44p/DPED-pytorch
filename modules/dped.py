import torch
import torch.nn as nn

from class_utils import import_class


class DPEDModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self._params_to_optim, self.generator, self.discriminator = self._prepare_models()
        self.criterion = self._prepare_criterion()
        self.optimizer = self._prepare_optimizer()
        
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
        discriminator.train(True)

    
        if self.config.trainer.get('resume_path'):
            load_model(self, self.config.trainer.resume_path)

        return params_to_optim, generator, discriminator
    
    def _prepare_criterion(self):
        return import_class(self.config.model.generator.criterion.get('module', torch.nn.MSELoss))(
                self,
                **self.config.model.generator.criterion.get("args", {'reduction': 'none'})
            )
    
    def _prepare_optimizer(self):
        params_to_optim = self._params_to_optim

        return import_class(self.config.hyperparameters.optimizer.name)(
            params_to_optim,
            **self.config.hyperparameters.optimizer.args
        )
    
    def forward(self, model_input, target):
        losses_mean = []

        output = self.generator(model_input)
        loss = self.criterion(output, target).mean()

        losses_mean.append(loss.item())

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return losses_mean
