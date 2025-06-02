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

        resume_path = self.config.trainer.get('resume_path')
        if resume_path:
            load_model(self.model, resume_path, device=self.device)
            print(f"Loaded the checkpoint from {resume_path}!")
        
        self.end_epoch = self.config.trainer.get("end_epoch", 1)
        self.end_step = self.config.trainer.get("end_step", 0)

        self.checkpoint_step = self.config.trainer.get("checkpoint_step", 0)
        self.checkpoint_epoch = self.config.trainer.get("checkpoint_epoch", 0)
        self.checkpoint_name = self.config.trainer.get("checkpoint_name", "checkpoint")

        self.eval_steps = self.config.evaluation.get("frequency_steps", 0)
        self.eval_epochs = self.config.evaluation.get("frequency_epochs", 0)
        self.eval_batch_size = self.config.evaluation.get("batch_size", 1)

        self.dataloader = self.prepare_dataloader()

        self.evaluators = self.prepare_evaluators()

        self.model = import_class(self.config.model.module)(self.config, self.device, self.evaluators)

        self.use_wandb = self.config.evaluation.get('use_wandb', False)
        if self.use_wandb:
            self.wandb_run = self.init_wandb()

    def prepare_dataloader(self):
        dataset = import_class(self.config.dataset.module)(
            **self.config.dataset.args,
            batch_size = self.config.hyperparameters.batch_size,
        )
        return dataset.get_dataloader()
    
    def prepare_evaluation_dataloader(self):
        dataset = import_class(self.config.evaluation.dataset.module)(
            **self.config.evaluation.dataset.args,
            batch_size = self.config.evaluation.batch_size,
        )
        return dataset.get_dataloader()
    
    def prepare_evaluators(self):
        evaluators = []
        dataloader = self.prepare_evaluation_dataloader()

        for evaluator_str in self.config.evaluation.metrics:
            evaluator = import_class(evaluator_str)(
                dataloader,
                device=self.device
            )
            evaluators.append(evaluator)
        
        return evaluators

    def checkpoint(self, path: pathlib.Path = None):
        name = f'{self.checkpoint_name}-epoch-{self.current_epoch}-step-{self.global_step}.safetensors'
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
        print(f'\n\nModel saved: {save_path}!\n\n')
    
    def init_wandb(self):
        return wandb.init(
            project=self.config.evaluation.wandb_project,
            config=OmegaConf.to_container(self.config)
        )
    
    def post_step(self, losses):
        if self.checkpoint_step and self.global_step > 0:
            if self.global_step % self.checkpoint_step == 0:
                self.checkpoint()

        metrics = {}

        if self.eval_steps and self.global_step > 0:
            if self.global_step % self.eval_steps == 0:
                metrics = self.eval()

        if self.use_wandb:
            report = {}
            for k, loss in losses.items():
                report[f"train/{k}"] = loss

            for k, metric in metrics.items():
                report[f"eval_step/{k}"] = metric

            self.wandb_run.log(report)

        if self.end_step:
            if self.global_step >= self.end_step:
                self.do_train = False
    
    def post_epoch(self):
        if self.checkpoint_epoch:
            if (self.current_epoch + 1) % self.checkpoint_epoch == 0:
                self.checkpoint()

        metrics = {}

        if self.eval_epochs:
            if (self.current_epoch + 1) % self.eval_epochs == 0:
                metrics = self.eval()

        if self.use_wandb:
            report = {}
            for k, metric in metrics.items():
                report[f"eval_epoch/{k}"] = metric

            if report:
                self.wandb_run.log(report)

        if self.end_epoch:
            if (self.current_epoch + 1) >= self.end_epoch:
                self.do_train = False
    
    def eval(self):
        metrics = {}

        torch.cuda.empty_cache()

        print()
        eval_bar = tqdm(self.evaluators, desc='Running evaluation...')
        for evaluator in eval_bar:
            eval_bar.set_description(f"{evaluator.name}")
            metrics[evaluator.name] = evaluator(self.model)
        
            torch.cuda.empty_cache()
        
        print("\n\n")
        print(' '.join(f"{name}: {metric}" for name, metric in metrics.items()))
        print()

        return metrics
    
    def end(self):
        save_path = self.config.trainer.get("save_path", "checkpoint.safetensors")
        if save_path:
            self.save_model(save_path)
        
        if self.use_wandb:
            wandb.finish()

    def train(self):
        self.current_epoch = 0
        self.global_step = 0

        self.do_train = True
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

                stat_str = ' '.join([f'{k} {loss:0.4f}' for k, loss in losses.items()])
                step_bar.set_postfix_str(stat_str)

                self.post_step(losses)

                if not self.do_train:
                    break
                
                self.global_step += 1

            self.post_epoch()

            if not self.do_train:
                break
            
            self.current_epoch += 1

        self.end()
