from class_utils import import_class

from abc import abstractmethod


class Evaluator:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    
    def __call__(self, model) -> float:
        return self.eval(model)
    
    @abstractmethod
    def eval(self, model) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def eval_batch(self, model, model_input, target) -> float:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplementedError
    
