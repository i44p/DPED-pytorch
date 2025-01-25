import torch
import pathlib
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from safetensors.torch import save_model, load_model
from modules import DPEDGenerator
from loss import DPEDLoss


def import_class(name=None):
    import importlib
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


class Trainer:
    def __init__(self, config: OmegaConf):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = config
        self.model = self.prepare_models()

        models = [self.model]
        self.optimizer = self.prepare_optimizer(models)
        self.dataloader = self.prepare_dataloader()

    def prepare_models(self):
        model = DPEDGenerator().to(self.device)
        trainer = self.config.trainer
        if trainer.get('resume_path'):
            load_model(model, traner.resume_path)
        model.train(True)
        return model

    def prepare_optimizer(self, models):
        params_to_optim = []

        for model in models:
            params_to_optim.append(
                {
                    "params": list(self.model.parameters())
                }
            )

        return import_class(self.config.hyperparameters.optimizer.name)(
            params_to_optim,
            **self.config.hyperparameters.optimizer.args
        )

    def prepare_dataloader(self):
        dataset = import_class(self.config.dataset.module)(
            **self.config.dataset.args,
            config=self.config,
        )
        return dataset.get_dataloader()

    def checkpoint(self, path: pathlib.Path = None):
        name = f'checkpoint-epoch-{self.current_epoch}-step-{self.global_step}.safetensors'
        if path:
            path = pathlib.Path(path)
        else:
            path = pathlib.Path(self.config.trainer.get('checkpoint_path', 'checkpoints/'))
        save_path = path / name
        self.save_model(save_path)

    def save_model(self, path: pathlib.Path):
        save_path = pathlib.Path(path).resolve()
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_model(self.model, save_path)
        print(f'Model saved: {save_path}!')

    def train(self):

        criterion = DPEDLoss()

        self.end_epoch = self.config.trainer.get("end_epoch", 1)
        self.end_step = self.config.trainer.get("end_step", 0)

        checkpoint_step = self.config.trainer.get("checkpoint_step", 0)
        checkpoint_epoch = self.config.trainer.get("checkpoint_epoch", 0)

        self.current_epoch = -1
        self.global_step = -1

        do_train = True
        epoch_bar = trange(1, self.end_epoch+1)
        while do_train:
            self.current_epoch += 1
            epoch_bar.n = self.current_epoch
            epoch_bar.set_description(f"Epoch: {self.current_epoch}")

            torch.cuda.empty_cache()

            step_bar = tqdm(self.dataloader, desc='step')
            for batch_idx, batch in enumerate(step_bar):
                self.global_step += 1
                step_bar.set_description(f"Global step: {self.global_step}")
                
                ###

                model_input = batch[0].to(self.device)
                target = batch[1].to(self.device)

                model_output = self.model(model_input)

                loss = criterion(model_output, target)

                loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                ###

                stat_str = f"loss: {loss:.4f}"
                step_bar.set_postfix_str(stat_str)

                if checkpoint_step and self.global_step > 0:
                    if self.global_step % checkpoint_step == 0:
                        self.checkpoint()

                if self.end_step:
                    if self.global_step >= self.end_step:
                        do_train = False
                        break

            if checkpoint_epoch and self.current_epoch > 0:
                if self.current_epoch % checkpoint_epoch == 0:
                    self.checkpoint()

            if self.end_epoch:
                if self.current_epoch >= self.end_epoch:
                    do_train = False
                    break

        save_path = self.config.trainer.get("save_path", "")
        if save_path:
            self.save_model(save_path)
