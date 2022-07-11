"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import numpy as np
import os

# Internal imports
import config.config as cfg

from data_loader.pytorch.base_pytorch_data_set import PytorchDataSet
from data_loader.pipeline.image.base import Float32er, SimpleInputScaler, UInt8er
from data_loader.pipeline.image.shape import ChannelLastToFirst, Cropper
from data_loader.pipeline.image.augmentations import color_augmentations, CopyPasteAugmenter, spatial_augmentations
from tools.io.io_utils import read_image


class DataSet(PytorchDataSet):
    """
        Pytorch data set, this class is responsible for loading in and pre-processing data given a list of file paths.
        This specific data set is for loading in images and corresponding segmentation files, with the image being a
        image format and the segmentation file a numpy binary.
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

        assert len(self.images) == len(self.segmentations), "Amount of images does not equal amount of segmentations"

        # Pipeline augmentations
        self.copy_paste_augmenter = CopyPasteAugmenter()
        self.cropper = Cropper((cfg.DATA_LOADER.IMAGE_SIZE, cfg.DATA_LOADER.IMAGE_SIZE), center=not self.training)
        self.spatial_augmentations = spatial_augmentations(probability_rotations_90=0.75)
        self.uint_eighter = UInt8er()
        self.image_color_augmentations = color_augmentations()
        self.float_thrity_twoer = Float32er()
        self.simple_input_scaler = SimpleInputScaler()
        self.channel_first = ChannelLastToFirst()

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
        # Unpack index if obtained by double sampler
        copy_paste_index = None
        if isinstance(index, tuple):
            index, copy_paste_index = index

        # Get the file name
        file_name = self.images[index].split(".")[0].split(os.path.sep)[-1]

        # Read
        image_path = self.images[index]
        segmentation_path = self.segmentations[index]
        image = read_image(image_path)
        segmentation = read_image(segmentation_path)
        if len(segmentation.shape) == 2:
            segmentation = segmentation[..., np.newaxis]
        else:
            segmentation = segmentation[..., :1]

        image = self.float_thrity_twoer.transform(image)
        segmentation = self.float_thrity_twoer.transform(segmentation)

        # Copy Paste Data Augmentation
        if self.training and cfg.DATA_LOADER.COPY_PASTE_DATA_AUGMENTATION and copy_paste_index is not None:
            copy_paste_image_path = self.images[copy_paste_index]
            copy_paste_segmentation_path = self.segmentations[copy_paste_index]
            copy_paste_image = read_image(copy_paste_image_path)
            copy_paste_segmentation = read_image(copy_paste_segmentation_path)
            if len(copy_paste_segmentation.shape) == 2:
                copy_paste_segmentation = copy_paste_segmentation[..., np.newaxis]
            else:
                copy_paste_segmentation = copy_paste_segmentation[..., :1]

            image, segmentation = self.copy_paste_augmenter.transform(copy_paste_image, copy_paste_segmentation,
                                                                      image, segmentation)

        # Resize
        composition = np.concatenate([image, segmentation], axis=2)
        if composition.shape[0] < cfg.DATA_LOADER.IMAGE_SIZE:
            _composition = np.zeros((cfg.DATA_LOADER.IMAGE_SIZE, composition.shape[1], composition.shape[2]),
                                    dtype=np.float32)
            _composition[:composition.shape[0]] = composition
            composition = _composition
        if composition.shape[1] < cfg.DATA_LOADER.IMAGE_SIZE:
            _composition = np.zeros((composition.shape[0], cfg.DATA_LOADER.IMAGE_SIZE, composition.shape[2]),
                                    dtype=np.float32)
            _composition[:, :composition.shape[1]] = composition
            composition = _composition
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
        image = self.simple_input_scaler.transform(image)

        # Convert to channel last
        image = self.channel_first.transform(image)
        segmentation = self.channel_first.transform(segmentation)

        # Create meta data file
        meta_data = {"file_name": file_name}

        return (image, segmentation), meta_data

    def __len__(self):
        """
        Get the length of the data set, check if there are as many images as segmentations first
        Returns: an int representing the length of the data set

        """
        length = len(self.images)
        return length
