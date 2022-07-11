"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
from torch.utils import data as torch_data

# Internal imports
import config.config as cfg
from data_loader.pytorch.base_pytorch_data_loader import PytorchDataLoader
from .inference_data_set import InferenceDataSet
from .data_set import DataSet
from data_loader.pytorch.samplers.limited_sampler import LimitedSampler
from data_loader.builder import DATA_LOADERS


@DATA_LOADERS.register_module(name='PYTORCH_IMAGE_MASK_SEGMENTATION')
class ImageMaskSegmentationDataLoader(PytorchDataLoader):
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
            prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR if hasattr(cfg.DATA_LOADER, "PREFETCH_FACTOR") else 10,
            sampler=LimitedSampler(train_data_set)
        )
        validation_data_set = DataSet(self.validation_file_paths, training=False)
        self.validation_data_loader = torch_data.DataLoader(
            validation_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR if hasattr(cfg.DATA_LOADER, "PREFETCH_FACTOR") else 10,
            collate_fn=self.collate_fn,
        )
        self.test_data_loader = torch_data.DataLoader(
            InferenceDataSet(self.test_file_paths, training=False),
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR if hasattr(cfg.DATA_LOADER, "PREFETCH_FACTOR") else 10,
            collate_fn=self.collate_fn,
        )
