import torch
import pathlib
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

    def prepare_dataloader(self):
        dataset = import_class(self.config.dataset.module)(
            **self.config.dataset.args,
            config=self.config,
        )
        return dataset.get_dataloader()

    def checkpoint(self, path: pathlib.Path):
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

    def train(self):
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
                
                losses = self.model(model_input, target)

                losses = [round(loss, 5) for loss in losses]

                ###

                stat_str = f"losses: {losses}"
                step_bar.set_postfix_str(stat_str)

                if checkpoint_step and self.global_step > 0:
                    if self.global_step % checkpoint_step == 0:
                        self.checkpoint()

                if self.end_step:
                    if self.global_step >= self.end_step:
                        do_train = False
                        break

            if checkpoint_epoch:
                if self.current_epoch % checkpoint_epoch == 0:
                    self.checkpoint()

            if self.end_epoch:
                if self.current_epoch >= self.end_epoch:
                    do_train = False
                    break

        save_path = self.config.trainer.get("save_path", "")
        if save_path:
            self.save_model(save_path)
