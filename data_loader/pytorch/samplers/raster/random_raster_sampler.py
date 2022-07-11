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


class RandomRasterSampler(Sampler):
    """
        Samples up from the data set up to a limit. This class should be used when working with large data sets and
        you want to shorten your epochs

    """

    def __init__(self, data_set):
        super().__init__(data_set)
        self.data_set = data_set

        self.number_of_instances = 0
        areas = []
        for size in self.data_set.raster_file_sizes:
            self.number_of_instances += size[0] * size[1] / cfg.DATA_LOADER.IMAGE_SIZE / cfg.DATA_LOADER.IMAGE_SIZE
            areas.append(size[0] * size[1])
        areas = np.array(areas)
        self.number_of_instances = int(self.number_of_instances)
        self.weights = areas / np.sum(areas)

        if hasattr(cfg.RUNNER, "MAX_ITERATIONS_PER_EPOCH"):
            self.length = min(cfg.RUNNER.MAX_ITERATIONS_PER_EPOCH * cfg.DATA_LOADER.BATCH_SIZE, self.number_of_instances)
        else:
            self.length = self.number_of_instances

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
