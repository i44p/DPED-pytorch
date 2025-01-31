import torch
import pathlib
import wandb
from class_utils import import_class
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from safetensors.torch import save_model, load_model


class Trainer:
    def __init__(self, config: OmegaConf):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = config

        self.model = import_class(self.config.model.module)(self.config, self.device)

        self.dataloader = self.prepare_dataloader()

        self.use_wandb = self.config.evaluation.get('use_wandb', False)
        if self.use_wandb:
            self.wandb_run = self.init_wandb()

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
        print()
        print(f'Model saved: {save_path}!')
    
    def init_wandb(self):
        return wandb.init(
            project=self.config.evaluation.wandb_project,
            config=dict(self.config)
        )

    def train(self):
        self.end_epoch = self.config.trainer.get("end_epoch", 1)
        self.end_step = self.config.trainer.get("end_step", 0)

        checkpoint_step = self.config.trainer.get("checkpoint_step", 0)
        checkpoint_epoch = self.config.trainer.get("checkpoint_epoch", 0)

        self.current_epoch = 0
        self.global_step = 0

        do_train = True
        epoch_bar = trange(self.end_epoch)
        for self.current_epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch: {self.current_epoch}")

            torch.cuda.empty_cache()

            step_bar = tqdm(self.dataloader, desc='step')
            for batch_idx, batch in enumerate(step_bar):
                step_bar.set_description(f"Global step: {self.global_step}")
                
                ###

                model_input = batch[0].to(self.device)
                target = batch[1].to(self.device)
                
                losses = self.model(model_input, target)

                ###

                stat_str = f"losses: {[round(loss, 5) for loss in losses]}"
                step_bar.set_postfix_str(stat_str)

                if checkpoint_step and self.global_step > 0:
                    if self.global_step % checkpoint_step == 0:
                        self.checkpoint()

                if self.use_wandb:
                    self.wandb_run.log({
                        "train/discriminator_loss": losses[0],
                        "train/generator_loss": losses[1]
                    })

                if self.end_step:
                    if self.global_step >= self.end_step:
                        do_train = False
                        break
                
                self.global_step += 1

            if checkpoint_epoch:
                if (self.current_epoch + 1) % checkpoint_epoch == 0:
                    self.checkpoint()

            if self.end_epoch:
                if (self.current_epoch + 1) >= self.end_epoch:
                    do_train = False
                    break
            
            if not do_train:
                break
            
            self.current_epoch += 1

        save_path = self.config.trainer.get("save_path", "")
        if save_path:
            self.save_model(save_path)
        
        if self.use_wandb:
            wandb.finish()
