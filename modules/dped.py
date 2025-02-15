import torch
import torch.nn as nn

from torchvision.transforms import Grayscale

from class_utils import import_class


class DPEDModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self.generator, self.discriminator = self._prepare_models()
        self.optimizer_generator, self.optimizer_discriminator = self._prepare_optimizers()
        self.criterion = self._prepare_criterion()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.grayscale = Grayscale()
        
    def _prepare_models(self):
        
        generator = import_class(self.config.model.generator.module)().to(self.device)
        generator.train(True)

        discriminator = import_class(self.config.model.discriminator.module)().to(self.device)
        discriminator.train(True)

        vgg = torch.hub.load('pytorch/vision', 'vgg19', pretrained=True).to(self.device)
        vgg.eval()
        vgg.requires_grad_(False)
        self.__dict__['vgg'] = vgg

        resume_path = self.config.trainer.get('resume_path')
        if resume_path:
            load_model(self, resume_path)
            print(f"Loaded the checkpoint from {resume_path}!")

        return generator, discriminator
    
    def _prepare_criterion(self):
        return import_class(self.config.model.generator.criterion.get('module', torch.nn.MSELoss))(
                **self.config.model.generator.criterion.get("args", {'reduction': 'none'})
            )
    
    def _prepare_optimizer(self, param_groups: list[dict]):
        return import_class(self.config.hyperparameters.optimizer.name)(
            param_groups,
            **self.config.hyperparameters.optimizer.args
        )

    def _prepare_optimizers(self):
        g_optim = self._prepare_optimizer([{
            "params": list(self.generator.parameters())
        }])

        d_optim = self._prepare_optimizer([{
            "params": list(self.discriminator.parameters()),
            "lr": float(self.config.hyperparameters.optimizer.args.lr) * float(self.config.model.discriminator.lr_factor),
        }])

        return g_optim, d_optim
    
    def forward(self, model_input, target):
        losses = {}
        
        discriminator_loss = self._discriminator_pass(model_input, target)
        losses['discrim'] = discriminator_loss

        generator_loss, other = self._generator_pass(model_input, target)
        losses['generator'] = generator_loss

        for k, loss in other.items():
            losses[k] = loss.mean().item()

        return losses
    
    def _discriminator_pass(self, model_input, target):
        with torch.no_grad():
            fake = self.grayscale(self.generator(model_input))
            real = self.grayscale(target)

        # branchless fake/real condition, mix per-image values
        batch = target.shape[0]
        target_prob = torch.randint(0, 2, [batch, 1], device=self.device).float()
        discriminator_input = fake * (1 - target_prob.view([batch, 1, 1, 1])) + \
                              real * target_prob.view([batch, 1, 1, 1])
                              
        discriminator_target = torch.cat([target_prob, 1-target_prob], 1)

        discriminator_output = self.discriminator(discriminator_input)

        loss_discriminator = self.cross_entropy(discriminator_output, discriminator_target)
        loss = loss_discriminator.mean()

        loss.backward()
        self.optimizer_discriminator.step()
        self.optimizer_discriminator.zero_grad(set_to_none=True)

        return loss.item()
    
    def _generator_pass(self, model_input, target):
        output = self.generator(model_input)
        generator_loss, other = self.criterion(output, target, self.discriminator, self.vgg)
        
        loss = generator_loss.mean()

        loss.backward()
        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad(set_to_none=True)

        return loss.item(), other
