"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod

# Internal imports
import config.config as cfg
from data_loader.base_data_loader import BaseDataLoader
import tools.data.type as type_converter


class PytorchDataLoader(BaseDataLoader, ABC):
    """
        Pytorch data loader creates 3 separate pytorch data loaders, one for each split (train, validation, test)
    """

    @abstractmethod
    def __init__(self, data_set_configurations):
        """
        Initialise the data loader

        Args:
            data_set_configurations: dictionary with key name of the data set and value data_set_configuration
        """
        super().__init__(data_set_configurations)

        # Get the data set module from the config
        self.data_function = getattr(type_converter, cfg.MODEL.INPUT_FUNCTION)

        # Data sets
        self.train_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None

    def collate_fn(self, data):
        """
        How to collate the loaded data

        Args:
           data: is a list of tuples with ((example, label, ...), meta_data_dictionary)
        """
        # Create variables
        _data, _meta_data = [], []
        for data_entry, meta_data_entry in data:
            _data.append(data_entry)
            _meta_data.append(meta_data_entry)

        data = self.stack_data(_data)
        meta_data = self.stack_meta_data(_meta_data)

        # Convert to the required data type
        data = [self.data_function(d) if not all(d_entry is None for d_entry in d) else None for d in data]

        return data, meta_data

    def stack_data(self, _data):
        """
        Stack data for batch usage

        Args:
            _data: list containing data tuples

        Returns: a tuple of lists containing data
        """
        # Setup lists
        data = [[] for _ in range(len(_data[0]))]

        # Insert data
        for index_tuple, data_tuple in enumerate(_data):
            for index_element, data_element in enumerate(data_tuple):
                data[index_element].append(data_element)

        return data

    def stack_meta_data(self, _meta_data):
        """
        Stack meta_data for batch usage

        Args:
            _meta_data: list containing dictionaries

        Returns: a single dictionary
        """
        # Setup meta data dictionary
        meta_data = {"batch_size": len(_meta_data)}

        if "stackable_keys" in _meta_data[0]:
            stackable_keys = _meta_data[0]["stackable_keys"]
            for key in stackable_keys:
                meta_data[key] = []
        else:
            stackable_keys = []

        for meta_data_dictionary in _meta_data:
            for key, value in meta_data_dictionary.items():
                if key not in meta_data:
                    meta_data[key] = value
                elif key in stackable_keys:
                    meta_data[key].append(value)

        return meta_data
