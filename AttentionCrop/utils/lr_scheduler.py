import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

class LR_Lambda:
    def __init__(self, cfg) -> None:
        lr = cfg['train']['optimizer']['lr']
        self.total_epochs = cfg['train']['total_iterations']
        self.warmup_epochs = cfg['train']['lr_scheduler']['warmup_iterations']
        self.lr_warmup = cfg['train']['lr_scheduler']['warmup_lr'] / lr
        self.lr_min = cfg['train']['lr_scheduler']['min_lr'] / lr
        self.lr_init = 1

    def __call__(self, epoch):
        if epoch <= self.warmup_epochs:
            return epoch * (self.lr_init-self.lr_warmup) / self.warmup_epochs + self.lr_warmup
        else:
            return (self.lr_init-self.lr_min) * np.cos((np.pi/2)*(epoch-self.warmup_epochs)/(self.total_epochs-1-self.warmup_epochs)) + self.lr_min

def create_scheduler(cfg, optimizer):
    lr_lambda = LR_Lambda(cfg)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler