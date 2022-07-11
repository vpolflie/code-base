"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """
        Warm up scheduler, updates the learning rate in a linear fashion from 0 -> 1.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.progress = 0
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.progress for base_lr in self.base_lrs]

    def step(self, progress=0.0):
        self.progress = progress
        super().step()
