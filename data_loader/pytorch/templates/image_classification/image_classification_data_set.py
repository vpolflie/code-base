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
from data_loader.pipeline.image.shape import Cropper, ChannelLastToFirst
from data_loader.pipeline.image.augmentations import color_augmentations, spatial_augmentations
from tools.io.io_utils import read_image, read_json


class DataSet(PytorchDataSet):
    """
        Pytorch data set, this class is responsible for loading in and pre-processing data given a list of file paths.
        This specific data set is for loading in images and corresponding categories, with the image being a
        image format and the category being a json.
    """

    def __init__(self, files_dictionary, training=True):
        """
        Initialise the data set by assigning the file path to the correct variable and initialising the preprocessing
        pipelines.

        Args:
            files_dictionary: a dictionary of files where the key is the type of the files and the key is a list of files
            training: whether or not this is a inference data set
        """
        super().__init__(files_dictionary, training)
        # Files should have the same naming convention so sorting the list should sync them as well
        self.images = np.array(sorted(files_dictionary["images"]))
        self.annotations = np.array(sorted(files_dictionary["annotations"]))

        assert len(self.images) == len(self.annotations), "Amount of images does not equal amount of segmentations"

        if len(self.images) > 0:
            # Extract annotations
            annotations = []
            for file_path in self.annotations:
                annotations.append(read_json(file_path)["material"])
            categories = np.unique(annotations)

            assert len(categories) == cfg.DATA_LOADER.NUMBER_OF_CLASSES, \
                "Amount of categories (%d) doesn't match the config 'number_of_classes' (%d) parameter" % \
                (len(categories), cfg.DATA_LOADER.NUMBER_OF_CLASSES)

            _annotations = np.zeros((len(annotations), len(categories)), dtype=np.float32)
            for i in range(len(categories)):
                _annotations[np.where(categories[i] == np.array(annotations)), i] = 1
            self.annotations = _annotations

        # Pipeline augmentations
        self.cropper = Cropper((cfg.DATA_LOADER.IMAGE_SIZE[0], cfg.DATA_LOADER.IMAGE_SIZE[1]), center=True)
        self.spatial_augmentations = spatial_augmentations()
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
            a numpy array of pre-processed image and a numpy array of a pre-processed segmentation
        """
        file_name = self.images[index].split(".")[0].split(os.path.sep)[-1]

        # Read
        image_path = self.images[index]
        annotation = self.annotations[index]
        image = read_image(image_path)
        image = self.float_thrity_twoer.transform(image)

        # Resize
        image = self.cropper.transform(image)

        # Augment
        if self.training:
            image = self.spatial_augmentations(image=image)["image"]

            # Set correct type
            image = self.uint_eighter.transform(image)
            image = self.image_color_augmentations(image=image)["image"]
            image = self.float_thrity_twoer.transform(image)
        image = self.simple_input_scaler.transform(image)
        image = self.channel_first.transform(image)

        # Create meta data file
        meta_data = {"file_name": file_name}

        return (image, annotation), meta_data

    def __len__(self):
        """
        Get the length of the data set, check if there are as many images as segmentations first
        Returns: an int representing the length of the data set

        """
        length = len(self.images)
        return length

    def get_class_balance(self):
        """
        Getter function which returns a class balance array
        """
        return np.sum(self.annotations, axis=0)

    def get_classes(self):
        """
        Getter function which returns a class balance array
        """
        return self.annotations
