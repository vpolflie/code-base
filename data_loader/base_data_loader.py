"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External Imports
from abc import ABC

# Internal imports
import config.config as cfg


class BaseDataLoader(ABC):
    """
        Base abstract class for a data loader, library specific data loaders should inherit from this class and provide
        generators in the train_data_loader, validation_data_loader and test_data_loader variables.
    """

    def __init__(self, data_set_configurations):
        """
        Initialise a data loader

        Args:
            data_set_configurations: a dictionary of containing the name of data set as key and a data configuration
            objects as value
        """
        self.data_set_configurations = data_set_configurations
        self.train_file_paths = {}
        self.validation_file_paths = {}
        self.test_file_paths = {}

        # For each data configuration extract per split (train, validation ans test) the corresponding files
        keys = list(data_set_configurations.items())[0][1].config["file_types"]
        for split in ["train", "validation", "test"]:
            split_file_paths = split + "_file_paths"
            setattr(self, split_file_paths, {k: [] for k in keys})
            for name, data_set_configuration in self.data_set_configurations.items():
                if data_set_configuration.config["name"] in getattr(cfg.DATA_LOADER, ("data_sets_" + split).upper()):
                    for k, v in getattr(data_set_configuration, split_file_paths).items():
                        current_dictionary = getattr(self, split_file_paths)
                        if k in current_dictionary:
                            current_dictionary[k] += v
                        else:
                            current_dictionary[k] = v
                        setattr(self, split_file_paths, current_dictionary)

        # Initialise the variables
        self.train_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None

    def get_train_data_loader(self):
        """
        A getter function to get the training data generator.

        Returns:
            A generator which loops over the training data

        """
        return self.train_data_loader

    def get_validation_data_loader(self):
        """
        A getter function to get the validation data generator.

        Returns:
            A generator which loops over the validation data

        """
        return self.validation_data_loader

    def get_test_data_loader(self):
        """
        A getter function to get the testing data generator.

        Returns:
            A generator which loops over the testing data

        """
        return self.test_data_loader
