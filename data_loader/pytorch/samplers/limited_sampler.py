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


class LimitedSampler(Sampler):
    """
        Samples up from the data set up to a limit. This class should be used when working with large data sets and
        you want to shorten your epochs
    """

    def __init__(self, data_set):
        super().__init__(data_set)
        self.data_set = data_set
        self.probability = np.ones(len(self.data_set))

        if hasattr(cfg.RUNNER, "MAX_ITERATIONS_PER_EPOCH"):
            self.length = min(cfg.RUNNER.MAX_ITERATIONS_PER_EPOCH * cfg.DATA_LOADER.BATCH_SIZE, len(self.data_set))
        else:
            self.length = len(self.data_set)

    def __iter__(self):
        """
        This function needs to return an iterable over which the __getitem__ from a pytorch data set can iterate
        Returns:
            an iterable e.g. a list
        """
        if np.sum(self.probability < self.length):
            self.probability = np.ones(len(self.data_set))
        choices = np.random.choice(range(len(self.data_set)), self.length, False, p=self.probability / np.sum(self.probability))
        self.probability[choices] = 0
        return iter(choices)

    def __len__(self):
        """
        Returns the length of the iterable that gets return in __iter__
        Returns:
            integer representing the length of the sampler
        """
        return self.length
