"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod

# Internal imports


class ModelAPI(ABC):

    """
    This is the model API that each deep learning library implementation of a model should follow
    """

    def __init__(self):
        super().__init__()

        pass

    @abstractmethod
    def build(self):
        """
            Build the model
        """
        pass

    @abstractmethod
    def initialize_model(self):
        """
            Initialise the model
        """
        pass

    @abstractmethod
    def train(self, x):
        """
        Perform a training step on the model

        Args:
            x: data

        Returns:

        """
        pass

    @abstractmethod
    def evaluate(self, x, return_output=True):
        """
        Perform an evaluation step on the model
        Args:
            x: data

        Returns:

        """
        pass

    @abstractmethod
    def inference(self, x):
        """
        Perform an inference step using the model
        Args:
            x: data

        Returns:

        """
        pass

    @abstractmethod
    def save_model(self, epoch):
        """
        Save the model
        Args:
            epoch: current epoch

        Returns:

        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        Load a previously saved model
        Args:
            path: path to the saved model

        Returns:

        """
        pass

    @abstractmethod
    def get_visualisation_data(self):
        """
            Package up the data which needs to be sent to the visualiser
            :returns: a list of VisualisationData objects
        """
        pass

    @abstractmethod
    def update_optimizer_learning_rate(self, update_percentage):
        """
        Update the learning rate of the optimizers

        args:
            update_percentage: how much of the learning rate update we've done
        """
        pass

    @abstractmethod
    def update_optimizer_warm_up(self, update_percentage):
        """
        Update the learning rate of the optimizers in terms of warm up. This means slowly building up the learning
        rate from 0 to the required value so the optimizer can figure out it's gradients

        args:
            update_percentage: how much of the learning rate update we've done
        """
        pass

    @abstractmethod
    def apply_stochastic_weight_averaging(self):
        """
        Update the optimizer to use stochastic weight averaging
        """
        pass

    @abstractmethod
    def get_header_information(self):
        """
            Get the loss names of the network and additional information
            :returns: a tuple of a list of strings
        """
        pass
