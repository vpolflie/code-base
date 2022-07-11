"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports

from torch.optim.lr_scheduler import _LRScheduler, StepLR


class LinearLR(_LRScheduler):
    """
        Linear scheduler, updates the learning rate in a linear fashion for the progress throughout the training process
    """
    def __init__(self, optimizer, last_epoch=-1, min_lr=1e-6):
        self.min_lr = min_lr
        self.progress = 0
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.progress), self.min_lr) for base_lr in self.base_lrs]

    def step(self, progress=0.0):
        self.progress = progress
        super().step()
