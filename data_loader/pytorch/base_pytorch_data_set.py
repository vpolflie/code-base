"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod
from torch.utils import data as torch_data

# Internal imports


class PytorchDataSet(ABC, torch_data.Dataset):
    """
        Base Class for a pytorch data set
    """

    def __init__(self, files_dictionary, training=True):
        super().__init__()
        self.files_dictionary = files_dictionary
        self.training = training

    @abstractmethod
    def __getitem__(self, item):
        """
        Load in an augment data, note that all data is expected in channel first format
        Args:
            item:

        Returns:

        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_class_balance(self):
        """
        Getter function which returns a class balance array
        """
        raise NotImplementedError("The function '%s' has not been implemented" % "get_class_balance")

    def get_classes(self):
        """
        Getter function which returns a class balance array
        """
        return NotImplementedError("The function '%s' has not been implemented" % "get_classes")
