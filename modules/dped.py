import torch
import torch.nn as nn

from torchvision.transforms import Grayscale
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from safetensors.torch import load_model

from class_utils import import_class
from modules.eval.evaluator import Evaluator


class DPEDModel(nn.Module):
    def __init__(self, config, device, evaluators: list[Evaluator]):
        super().__init__()

        self.config = config
        self.device = device

        self.generator, self.discriminator = self._prepare_models()
        self.optimizer_generator, self.optimizer_discriminator = self._prepare_optimizers()
        self.criterion = self._prepare_criterion()
        
        self.evaluators = evaluators
        self.report_train_metrics = self.config.evaluation.get('report_train_metrics', False)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.grayscale = Grayscale()

        self.preprocessor = import_class(self.config.model.preprocessor.module)(
            **self.config.model.preprocessor.args
        )
        
    def _prepare_models(self):
        
        generator = import_class(self.config.model.generator.module)().to(self.device)
        generator.eval()

        discriminator = import_class(self.config.model.discriminator.module)().to(self.device)
        discriminator.eval()

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(self.device)
        vgg.eval()
        vgg.requires_grad_(False)
        feature_level = str(self.config.model.generator.get('vgg_feature_level', 35))
        self.__dict__['vgg'] = create_feature_extractor(vgg, return_nodes={'features.' + feature_level: 'feature_layer'})

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
        
        model_input = self.preprocessor.encode(model_input)
        target = self.preprocessor.encode(target)
        
        discriminator_loss = self._discriminator_pass(model_input, target)
        generator_loss, other = self._generator_pass(model_input, target)
        
        losses['discrim_loss'] = discriminator_loss.item()
        losses['generator_loss'] = generator_loss.item()

        if self.report_train_metrics:
            for evaluator in self.evaluators:
                losses[evaluator.name] = evaluator.eval_batch(self, model_input, target)

        for k, loss in other.items():
            losses[k] = loss.mean().item()

        return losses
    
    def _discriminator_pass(self, model_input, target):
        self.discriminator.train(True)
        self.discriminator.requires_grad_(True)
        self.generator.eval()
        self.generator.requires_grad_(False)

        real = self.grayscale(target)
        fake = self.grayscale(self.generator(model_input))
            

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

        return loss
    
    def _generator_pass(self, model_input, target):
        self.discriminator.eval()
        self.discriminator.requires_grad_(False)
        self.generator.train(True)
        self.generator.requires_grad_(True)

        output = self.generator(model_input)
        generator_loss, other = self.criterion(output, target, self.discriminator, self.vgg)
        
        loss = generator_loss.mean()

        loss.backward()
        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad(set_to_none=True)

        return loss, other
