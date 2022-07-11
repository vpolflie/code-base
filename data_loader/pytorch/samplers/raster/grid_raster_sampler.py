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


class GridRasterSampler(Sampler):
    """
        Samples up from the data set up to a limit. This class should be used when working with large data sets and
        you want to shorten your epochs

    """

    def __init__(self, data_set):
        super().__init__(data_set)
        self.data_set = data_set
        
        # Order the indices in such a manner that its write efficient
        self.indices = []
        bucket = {}
        for index, patch in enumerate(self.data_set.patches):
            patch_index = patch[0]
            if patch_index not in bucket:
                bucket[patch_index] = [index]
            else:
                bucket[patch_index] += [index]

        number_of_buckets = cfg.RUNNER.NUMBER_OF_WRITERS if hasattr(cfg.RUNNER, "NUMBER_OF_WRITERS") else 1
        last_index = min(len(bucket.keys()), number_of_buckets)
        writer_indices = [i for i in range(min(len(bucket.keys()), number_of_buckets))]
        counters = [0 for _ in range(min(len(bucket.keys()), number_of_buckets))]
        done = [False] * len(bucket.keys())

        while not all(done):
            for index_writer in range(min(len(bucket.keys()), number_of_buckets)):
                if len(bucket[writer_indices[index_writer]]) > counters[index_writer]:
                    self.indices.append(bucket[writer_indices[index_writer]][counters[index_writer]])
                    counters[index_writer] += 1
                else:
                    done[writer_indices[index_writer]] = True
                    if last_index < len(bucket.keys()):
                        counters[index_writer] = 0
                        writer_indices[index_writer] = last_index
                        last_index += 1

    def __iter__(self):
        """
        This function needs to return an iterable over which the __getitem__ from a pytorch data set can iterate
        Returns:
            an iterable e.g. a list
        """
        return iter(self.indices)

    def __len__(self):
        """
        Returns the length of the iterable that gets return in __iter__
        Returns:
            integer representing the length of the sampler
        """
        length = len(self.indices)
        return length
