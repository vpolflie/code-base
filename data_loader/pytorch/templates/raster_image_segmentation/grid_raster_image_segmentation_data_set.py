"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import numpy as np
import os
import rasterio

# Internal imports
import config.config as cfg

from data_loader.pytorch.base_pytorch_data_set import PytorchDataSet
from data_loader.pipeline.image.base import Float32er, SimpleInputScaler, UInt8er
from data_loader.pipeline.image.shape import ChannelLastToFirst, ChannelFirstToLast
from data_loader.pipeline.image.augmentations import color_augmentations, spatial_augmentations
from tools.io.raster.rasterio_utils import read_window_path


class GridDataSet(PytorchDataSet):
    """
        Pytorch data set, this class is responsible for loading in and pre-processing data given a list of file paths.
        This specific data set is for loading in images and corresponding segmentation files, with the image and the
        segmentation both being larger raster files. They will be iterated over in a grid manner.
    """

    def __init__(self, files_dictionary, training=True):
        """
        Initialise the data set by assigning the file path to the correct variable and initialising the preprocessing
        pipelines.

        Args:
            files_dictionary: a dictionary of files where the key is the type of the files and the key is a list of files
            training: whether or not this is a training data set
        """
        super().__init__(files_dictionary, training)
        # Files should have the same naming convention so sorting the list should sync them as well
        self.images = np.array(sorted(files_dictionary["images"]))
        self.segmentations = np.array(sorted(files_dictionary["segmentations"]))

        # Get all the patches
        self.patches = []
        for index in range(len(self.images)):
            data_file = rasterio.open(self.images[index])
            size = data_file.shape
            data_file.close()

            # Loop over all possible examples
            for i in range(-cfg.DATA_LOADER.OVERLAP, size[0], cfg.DATA_LOADER.IMAGE_SIZE - 2 * cfg.DATA_LOADER.OVERLAP):
                for j in range(-cfg.DATA_LOADER.OVERLAP, size[1], cfg.DATA_LOADER.IMAGE_SIZE - 2 * cfg.DATA_LOADER.OVERLAP):
                    self.patches.append((index,
                                         i,
                                         j,
                                         cfg.DATA_LOADER.IMAGE_SIZE,
                                         cfg.DATA_LOADER.IMAGE_SIZE))

        # Pipeline augmentations
        self.spatial_augmentations = spatial_augmentations()
        self.uint_eighter = UInt8er()
        self.image_color_augmentations = color_augmentations()
        self.float_thrity_twoer = Float32er()
        self.simple_input_scaler = SimpleInputScaler()
        self.channel_first = ChannelLastToFirst()
        self.channel_last = ChannelFirstToLast()

    def __getitem__(self, index):
        """
        Prepare an image - segmentation pair. And apply augmentations to it:

        image -> spatial augmentation -> type: uint8 -> color augmentation -> type: float_32 -> rescale to [0-1]
        segmentation -> spatial augmentation -> type: float_32

        Args:
            index: integer pointing to the index of the image and segmentation list

        Returns:
            a tuple of numpy arrays of pre-processed image and a numpy array of a pre-processed segmentation and the
            corresponding filename
        """
        # Get the file name
        file_index, x, y, width, height = self.patches[index]
        image_path = self.images[file_index]
        file_name = image_path.split(".")[0].split(os.path.sep)[-1]

        # Read Image
        image, meta_data = read_window_path(image_path, x, y, width, height)
        image = self.channel_last.transform(image)
        image = self.float_thrity_twoer.transform(image)

        # Read segmentations
        segmentation = None
        if len(self.segmentations):
            segmentation_path = self.segmentations[file_index]
            segmentation, _ = read_window_path(segmentation_path, x, y, width, height)
            segmentation = self.channel_last.transform(segmentation)
            segmentation = self.float_thrity_twoer.transform(segmentation)

            # Augment
            if self.training:
                composition = self.spatial_augmentations(image=np.concatenate([image, segmentation], axis=2))["image"]
                image, segmentation = composition[..., :3], composition[..., 3:]

                # Set correct type
                image = self.uint_eighter.transform(image)
                image = self.image_color_augmentations(image=image)["image"]
                image = self.float_thrity_twoer.transform(image)

            segmentation = self.channel_first.transform(segmentation)
        image = self.simple_input_scaler.transform(image)
        image = self.channel_first.transform(image)

        # Create meta data file
        meta_data["file_name"] = file_name
        meta_data["file_index"] = file_index
        meta_data["overlap"] = cfg.DATA_LOADER.OVERLAP

        return (image, segmentation), meta_data

    def __len__(self):
        """
        Get the length of the data set
        Returns: an int representing the length of the data set

        """
        length = len(self.patches)
        return length
