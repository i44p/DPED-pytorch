import torch
import torch.nn as nn

from torchvision.transforms import Grayscale

from class_utils import import_class


class DPEDModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self._params_to_optim, self.generator, self.discriminator = self._prepare_models()
        self.criterion = self._prepare_criterion()
        self.optimizer = self._prepare_optimizer()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.grayscale = Grayscale()
        
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
        
        discriminator_loss = self._discriminator_pass(model_input, target)
        losses_mean.append(discriminator_loss)

        generator_loss = self._generator_pass(model_input, target)
        losses_mean.append(generator_loss)

        return losses_mean
    
    def _discriminator_pass(self, model_input, target):
        # branchless condition, per-image loss

        batch = target.shape[0]

        target_prob = torch.randint(0,2,[batch, 1]).float()

        output = self.generator(model_input)
        grayscale_output = self.grayscale(output)

        discriminator_input = grayscale_output * (1 - target_prob.view([batch, 1, 1, 1])) + \
                              self.grayscale(target) * target_prob.view([batch, 1, 1, 1])
        discriminator_target = torch.cat([target_prob, 1-target_prob], 1)[:,0]

        discriminator_output = self.discriminator(grayscale_output)
        discriminator_real_confidence = discriminator_output[:,0]

        loss_discriminator = -self.cross_entropy(discriminator_real_confidence, discriminator_target)
        loss = loss_discriminator.mean()

        loss_discriminator.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss_discriminator.item()
    
    def _generator_pass(self, model_input, target):
        output = self.generator(model_input)
        loss = self.criterion(output, target).mean()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.item()
