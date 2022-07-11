"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
import os
import random
import rasterio
from skimage.transform import resize

# Internal imports
import config.config as cfg

from data_loader.pytorch.base_pytorch_data_set import PytorchDataSet
from data_loader.pipeline.general import Normalize
from data_loader.pipeline.image.base import Float32er, SimpleInputScaler, UInt8er, Int32er
from data_loader.pipeline.image.shape import ChannelLastToFirst, Cropper, Padder
from data_loader.pipeline.image.augmentations import color_augmentations, spatial_augmentations
from tools.io.raster.rasterio_utils import read_window_path


class DataSet(PytorchDataSet):
    """
        Pytorch data set, this class is responsible for loading in and pre-processing data given a list of file paths.
        This specific data set is for loading in images and corresponding segmentation files, with the image and segmentations
        both being of a larger raster file (tif/geotiff)
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

        # Get the size of the images
        self.raster_file_sizes = []
        for image_file in self.images:
            data_file = rasterio.open(image_file)
            self.raster_file_sizes.append(data_file.shape)
            data_file.close()

        assert len(self.images) == len(self.segmentations), "Amount of images does not equal amount of segmentations"

        # Pipeline augmentations
        self.cropper = Cropper((cfg.DATA_LOADER.IMAGE_SIZE, cfg.DATA_LOADER.IMAGE_SIZE), center=not self.training)
        self.padder = Padder(cfg.DATA_LOADER.IMAGE_SIZE, cfg.DATA_LOADER.IMAGE_SIZE, constant_values=0, mode="reflect")
        self.spatial_augmentations = spatial_augmentations(probability_rotations_90=0.75)
        self.uint_eighter = UInt8er()
        self.image_color_augmentations = color_augmentations()
        self.float_thrity_twoer = Float32er()
        self.simple_input_scaler = SimpleInputScaler()
        self.channel_first = ChannelLastToFirst()
        self.int_thirty_twoer = Int32er()
        self.normalizer = Normalize(mean=cfg.DATA_LOADER.MEAN, std=cfg.DATA_LOADER.STD)

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
        file_name = self.images[index].split(".")[0].split(os.path.sep)[-1]

        # Read
        # Read image and segmentation and get a random patch
        image_path = self.images[index]
        x = random.randint(0, self.raster_file_sizes[index][1] - cfg.DATA_LOADER.IMAGE_SIZE)
        y = random.randint(0, self.raster_file_sizes[index][0] - cfg.DATA_LOADER.IMAGE_SIZE)
        image, meta_data = read_window_path(image_path, x, y, cfg.DATA_LOADER.IMAGE_SIZE, cfg.DATA_LOADER.IMAGE_SIZE)

        # Transform to float
        image = self.float_thrity_twoer.transform(image)

        # Read segmentation
        segmentation_path = self.segmentations[index]
        segmentation, _ = read_window_path(segmentation_path, x, y, cfg.DATA_LOADER.IMAGE_SIZE, cfg.DATA_LOADER.IMAGE_SIZE)

        # Create one hot encoding
        _segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], cfg.DATA_LOADER.SEGMENTATION_CHANNELS))
        for index in range(cfg.DATA_LOADER.SEGMENTATION_CHANNELS):
            _segmentation[..., index] = segmentation[..., 0] == index
        segmentation = _segmentation

        # Resize to image shape
        segmentation = resize(segmentation, image.shape[:2], anti_aliasing=False, order=0)
        segmentation = self.float_thrity_twoer.transform(segmentation)

        # Resize
        composition = np.concatenate([image, segmentation], axis=2)
        composition = self.padder.transform(composition)
        composition = self.cropper.transform(composition)
        image, segmentation = composition[..., :3], composition[..., 3:]

        # Augment
        if self.training:
            composition = self.spatial_augmentations(image=np.concatenate([image, segmentation], axis=2))["image"]
            image, segmentation = composition[..., :3], composition[..., 3:]

            # Set correct type
            image = self.uint_eighter.transform(image)
            image = self.image_color_augmentations(image=image)["image"]
            image = self.float_thrity_twoer.transform(image)

        # Normalize
        image = self.simple_input_scaler.transform(image)
        if cfg.DATA_LOADER.NORMALIZE:
            image = self.normalizer.transform(image)

        # Set segmentation back to single channel
        segmentation = np.argmax(segmentation, axis=-1)[..., np.newaxis]
        segmentation = self.int_thirty_twoer.transform(segmentation)

        # Convert to channel last
        image = self.channel_first.transform(image)
        segmentation = self.channel_first.transform(segmentation)

        meta_data["file_name"] = file_name
        meta_data["stackable_keys"] = ["file_name"]
        meta_data["color_coding"] = np.array([[0, 0, 0], [255, 0, 0], [128, 128, 128], [192, 192, 192], [255, 0, 255],
                                  [0, 0, 255], [128, 0, 128], [128, 0, 0], [0, 255, 0], [255, 255, 0],
                                  [0, 255, 255], [128, 128, 0], [0, 128, 0], [0, 128, 128], [0, 0, 128],
                                  [255, 255, 255], ])
        meta_data["normalize"] = cfg.DATA_LOADER.NORMALIZE
        meta_data["mean"] = cfg.DATA_LOADER.MEAN
        meta_data["std"] = cfg.DATA_LOADER.STD
        meta_data["clip_value_min"] = 0
        meta_data["clip_value_max"] = 1
        meta_data["overlap"] = cfg.DATA_LOADER.OVERLAP

        return (image, segmentation), meta_data

    def __len__(self):
        """
        Get the length of the data set
        Returns: an int representing the length of the data set

        """
        length = len(self.images)
        return length
