import torch

# todo: implement loss as defined in DPED paper
def DPEDLoss():
    return torch.nn.MSELoss(reduction='none')