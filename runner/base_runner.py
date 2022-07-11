"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod

# Internal imports


class BaseRunner(ABC):

    """
    Base class for the runner module. Main function is the run function that needs to be implemented by each child class.
    """
    def __init__(self, data_loader, model, visualiser):
        """
        Save all the input variables as class variables
        Args:
            data_loader: a data loader object
            model: a model object
            visualiser: a visualiser object
        """
        self.data_loader = data_loader
        self.model = model
        self.visualiser = visualiser

    def initialise(self):
        """
        Initialise the runner
        Returns:

        """
        pass

    @abstractmethod
    def run(self):
        """
        Run through the logic
        Returns:

        """
        return

    @staticmethod
    def get_batch_size(data):
        """
        Get the batch size of a given data batch
        Args:
         data: a tuple of tensors or a single tensor

        Returns:
             an int representing the batch size
        """
        batch_size = max([len(data_entry) if data_entry is not None else 0 for data_entry in data])
        return batch_size
