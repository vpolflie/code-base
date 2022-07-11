"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from torch.utils import data as torch_data

# Internal imports
import config.config as cfg
from data_loader.pytorch.base_pytorch_data_loader import PytorchDataLoader
from data_loader.pytorch.samplers.raster.random_raster_sampler import RandomRasterSampler
from data_loader.pytorch.samplers.raster.grid_raster_sampler import GridRasterSampler
from data_loader.pytorch.templates.raster_image_segmentation.random_sample_raster_data_set import DataSet
from data_loader.pytorch.templates.raster_image_segmentation.grid_raster_data_set import GridDataSet
from data_loader.pytorch.templates.raster_image_segmentation.inference_grid_raster_data_set import InferenceGridDataSet
from data_loader.builder import DATA_LOADERS


@DATA_LOADERS.register_module(name='PYTORCH_RASTER_IMAGE_SEGMENTATION')
class PytorchRasterImageSegmentationDataLoader(PytorchDataLoader):
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
        train_data_set = DataSet(self.train_file_paths, training=True)
        self.train_data_loader = torch_data.DataLoader(
            train_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            collate_fn=self.collate_fn,
            prefetch_factor=10,
            sampler=RandomRasterSampler(train_data_set)
        )
        validation_data_set = GridDataSet(self.validation_file_paths, training=False)
        self.validation_data_loader = torch_data.DataLoader(
            validation_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            collate_fn=self.collate_fn,
            prefetch_factor=10,
            sampler=GridRasterSampler(validation_data_set)
        )

        test_data_set = InferenceGridDataSet(self.test_file_paths, training=False)
        self.test_data_loader = torch_data.DataLoader(
            test_data_set,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.WORKERS,
            collate_fn=self.collate_fn,
            prefetch_factor=10,
            sampler=GridRasterSampler(test_data_set)
        )

