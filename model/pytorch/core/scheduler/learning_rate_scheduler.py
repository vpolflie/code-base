"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from .warm_up_scheduler import WarmUpLR


class LearningRateScheduler:
    """
        Overarching class for the pytorch learning rate schedulers. This class contains 2 different schedulers.

        A warm up scheduler which can be used to warm up the gradients of the optimizer, this scheduler is used
        at the start of training and is updated after each iteration. When the progress of this scheduler reaches
        100%, the model achieved its base learning rate.
        A normal scheduler which can be used to update the learning rate during training after a set number of epochs

        Args:
            optimizer: pytorch optimizer
            scheduler_class: a scheduler_class which should be initialised

        Parameters:
            last_epoch: last epoch of the optimizer, useful when resuming training
            kwargs: additional parameters for the scheduler_class
    """

    def __init__(self, optimizer, scheduler_class, last_epoch=-1, warm_up=False, **kwargs):
        super().__init__()

        # Scheduler parameters
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Scheduler variables
        self.scheduler = scheduler_class(optimizer, last_epoch, **kwargs)
        if warm_up:
            self.warm_up_scheduler = WarmUpLR(optimizer, -1)

    def step(self, progress=0.0):
        """
            Method to step through the learning rate update progress

            Args:
                progress: float representing how much of the learning rate update has been completed
        """
        self.scheduler.step(progress)

    def step_warm_up(self, warm_up_progress=0.0):
        """
            Method to step through the warm up progress

            Args:
                warm_up_progress: float representing how much of the warm up has been completed
        """
        self.warm_up_scheduler.step(warm_up_progress)

    def set_last_epoch(self, last_epoch):
        """
            Set the last epoch of the scheduler to a new value, in case of resuming training/

            Args:
                last_epoch: integer, the new last epoch
        """
        self.scheduler.last_epoch = last_epoch
