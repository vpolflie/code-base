"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
from torch.utils.data.sampler import Sampler

# Internal imports
import config.config as cfg


class OverSampler(Sampler):
    """
        Samples up from the data but oversamples on certain classes to balance out the data set
    """

    def __init__(self, data_set):
        super().__init__(data_set)
        self.data_set = data_set
        self.classes = self.data_set.get_classes()
        self.class_balance = self.data_set.get_class_balance()
        if self.class_balance.size > 0:
            self.class_balance_percentages = self.class_balance / np.max(self.class_balance, keepdims=True)
            self.weights = np.sum(self.classes * 1 / self.class_balance_percentages, axis=1)
            self.weights /= np.sum(self.weights)

            self.length = int(np.sum(np.ceil(self.class_balance / self.class_balance_percentages)))
            if hasattr(cfg.RUNNER, "MAX_ITERATIONS_PER_EPOCH") and cfg.RUNNER.MAX_ITERATIONS_PER_EPOCH is not None:
                self.length = min(cfg.RUNNER.MAX_ITERATIONS_PER_EPOCH * cfg.DATA_LOADER.BATCH_SIZE, self.length)
        else:
            self.length = 0

    def __iter__(self):
        """
        This function needs to return an iterable over which the __getitem__ from a pytorch data set can iterate
        Returns:
            an iterable e.g. a list
        """
        return iter(np.random.choice(range(len(self.data_set)), self.length, True, self.weights))

    def __len__(self):
        """
        Returns the length of the iterable that gets return in __iter__
        Returns:
            integer representing the length of the sampler
        """
        return self.length
