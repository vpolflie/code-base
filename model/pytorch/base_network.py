"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod
import torch

# Internal imports


class BasePytorchNetwork(torch.nn.Module, ABC):

    """
        Base Network Class.
    """

    def __init__(self):
        """
            Base Network initialisation.
        """
        super().__init__()
        # State parameters
        self.evaluation = False

        # Device
        self.device = None

        # Setup Logger Info
        self.loss_names = []
        self.max_length = []
        self.comparison = []

        # Block/optimizers pairs for save/load
        self.network_blocks = []

    @abstractmethod
    def build(self):
        """
            Build the entire network using blocks and returns a dictionary of optimizers
        """
        pass

    def forward(self, x=None):
        """
            Base forward function that needs to be implemented by child network classes
        """
        self.set_input(x)

        if self.training:
            self.feed_forward()
            self.calculate_losses()
        else:
            with torch.no_grad():
                self.feed_forward()
                self.calculate_losses()

        return self.set_output()

    def inference(self, x):
        """
            Base forward function that needs to be implemented by child network classes
        """
        self.set_input(x)
        with torch.no_grad():
            self.feed_forward()

        return self.set_output()

    def train_state(self):
        """
        Set the network in evaluation state
        """
        self.train()
        self.evaluation = True

    def evaluation_state(self):
        """
        Set the network in evaluation state
        """
        self.eval()
        self.evaluation = True

    def inference_state(self):
        """
        Set the network in evaluation state
        """
        self.eval()
        self.evaluation = False

    @abstractmethod
    def set_input(self, x):
        """
            Set the input variables for the networks
            Args:
                x: input parameters
        """
        pass

    @abstractmethod
    def feed_forward(self):
        """
            Forward pass through the network
        """
        pass

    @abstractmethod
    def calculate_losses(self):
        """
            Calculate all the loss terms of the network
        """
        pass

    @abstractmethod
    def optimize_step(self, losses):
        """
            Optimize the networks over the corresponding losses
            Args:
                losses: list of losses to optimize over
        """
        pass

    @abstractmethod
    def set_output(self):
        """
            Set the output variables for the networks
            Returns: a tuple of output variables, first one being the results and the second on being the loss
        """
        pass

    @abstractmethod
    def update_learning_rate(self, update_percentage):
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
    def get_visualisation_data(self):
        """
        Get data for the visualisations
        Returns: A list of VisualisationData objects
        """
