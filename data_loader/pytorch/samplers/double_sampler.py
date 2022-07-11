"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np

# Internal imports
import config.config as cfg
from .limited_sampler import LimitedSampler


class DoubleSampler(LimitedSampler):
    """
        Samples up from the data set up to a limit. This class should be used when working with large data sets and
        you want to shorten your epochs
    """

    def __init__(self, data_set):
        super().__init__(data_set)

    def __iter__(self):
        """
        This function needs to return an iterable over which the __getitem__ from a pytorch data set can iterate
        Returns:
            an iterable e.g. a list
        """
        iterator_1 = super().__iter__()
        second_indices = np.random.choice(range(len(self.data_set)), self.length, False)
        return iter(zip(list(iterator_1), second_indices))
