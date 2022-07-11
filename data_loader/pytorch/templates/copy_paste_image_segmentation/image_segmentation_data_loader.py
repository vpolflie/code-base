"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import numpy as np
from torch.utils import data as torch_data

# Internal imports
import config.config as cfg
from data_loader.pytorch.base_pytorch_data_loader import PytorchDataLoader
from data_loader.pytorch.templates.copy_paste_image_segmentation.image_segmentation_data_set import DataSet
from data_loader.pytorch.samplers.double_sampler import DoubleSampler
from data_loader.pytorch.samplers.limited_sampler import LimitedSampler
from data_loader.builder import DATA_LOADERS


@DATA_LOADERS.register_module(name='PYTORCH_IMAGE_SEGMENTATION_COPY_PASTE')
class PytorchImageSegmentationCopyPasteDataLoader(PytorchDataLoader):
    """
        Pytorch data loader creates 3 separate pytorch data loaders, one for each split (train, validation, test)
    """

    def __init__(self, data_set_configurations):
        """
        Initialise the data loader

        Args:
            data_set_configurations: dictionary with key name of the data set and value data_set_configuration
        """
        super().__init__(data_set_configurations)

        # Initialise
        train_data_set = DataSet(self.train_file_paths)
        self.train_data_loader = torch_data.DataLoader(
            train_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            collate_fn=self.collate_fn,
            sampler=DoubleSampler(train_data_set)
        )
        validation_data_set = DataSet(self.validation_file_paths, training=False)
        self.validation_data_loader = torch_data.DataLoader(
            validation_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            collate_fn=self.collate_fn,
            sampler=LimitedSampler(validation_data_set)
        )
        self.test_data_loader = torch_data.DataLoader(
            DataSet(self.test_file_paths, training=False),
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            collate_fn=self.collate_fn,
            num_workers=cfg.DATA_LOADER.WORKERS,
        )

    def collate_fn(self, data):
        """
        How to collate the loaded data

        Args:
           data: is a list of tuples with ((example, label, ...), filename)
        """
        # Create variables
        _data = []
        file_names = []

        # Create lists
        if len(data) > 0:
            _data = [[] for _ in range(len(data[0][0]))]

        # Fill those lists
        for index in range(len(data)):
            actual_data = data[index][0]

            if np.all([np.sum(actual_data[i]) > 0 for i in range(len(actual_data))]):
                for data_index in range(len(actual_data)):
                    _data[data_index].append(actual_data[data_index])

                file_names.append(data[index][1])

        # Convert to float tensors
        _data = [self.data_function(d) for d in _data]
        return _data, file_names
