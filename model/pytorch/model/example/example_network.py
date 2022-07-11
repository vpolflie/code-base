"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch

# Internal imports
import config.config as cfg
from model.pytorch.base_network import BasePytorchNetwork
from model.pytorch.model.example.example_block import ExampleBlock
from visualiser.visualisation_data import VisualisationData


class ExampleNetwork(BasePytorchNetwork):

    """
        Example Network Class.
    """

    def __init__(self):
        """
            Base Model initialisation.
        """
        super().__init__()

        # Setup Logger Info
        self.loss_names = ["Softmax Cross Entropy"]
        self.max_length = ["%0.7f"]
        self.comparison = ["<"]

        # Block/optimizers pairs for save/load
        self.network_blocks = []

        # Network Variables
        self.auto_encoder = None
        self.auto_encoder_optimizer = None

        self.images = None
        self.segmentations = None
        self.prediction_segmentations = None

        self.loss = None
        self.loss_criterion = torch.nn.MSELoss()
    
    def build(self):
        """
            Build the entire network using blocks and returns a dictionary of optimizers
        """
        self.auto_encoder = ExampleBlock()
        self.auto_encoder.build()
        self.auto_encoder_optimizer = torch.optim.Adam(self.auto_encoder.parameters(), lr=cfg.MODEL.LEARNING_RATE)
        self.network_blocks.append((self.auto_encoder, self.auto_encoder_optimizer))

    def set_input(self, x):
        """
            Set the input variables for the networks
            Args:
                x: input parameters
        """
        self.images = x[0].to(self.device)
        self.segmentations = x[1].to(self.device)

    def feed_forward(self):
        """
            Forward pass through the network
        """
        self.prediction_segmentations = self.auto_encoder(self.images)

    def calculate_losses(self):
        """
            Calculate all the loss terms of the network
        """
        self.loss = self.loss_criterion(self.prediction_segmentations.reshape((-1, cfg.MODEL.OUTPUT_CHANNELS)),
                                        self.segmentations.reshape((-1, cfg.MODEL.OUTPUT_CHANNELS)))

    def optimize_step(self, losses):
        """
            Optimize the networks over the corresponding losses
            Args:
                losses: list of losses to optimize over
        """
        self.loss.backward()
        self.auto_encoder_optimizer.step()
    
    def set_output(self):
        """
            Set the output variables for the networks
            Returns: two tuples of output variables, first one being the results and the second on being the loss
        """
        return (self.prediction_segmentations,), (self.loss,)

    def update_learning_rate(self, epochs_left):
        """
        Update the learning rate of the optimizers

        args:
            epochs_left: the amount of epochs still left to train
        """
        pass

    def apply_stochastic_weight_averaging(self):
        """
        Update the optimizer to use stochastic weight averaging
        """
        pass

    def get_visualisation_data(self):
        """
        Get data for the visualisations
        Returns:
            A list of VisualisationData objects
        """
        data = [
            VisualisationData(
                "images", "Image", "Input Images",
                self.images.detach().cpu() if self.images is not None else None),
            VisualisationData(
                "images", "Segmentation", "Ground truth Segmentations",
                self.segmentations.detach().cpu() if self.segmentations is not None else None),
            VisualisationData(
                "images", "Predictions", "Predicted Segmentations",
                self.prediction_segmentations.detach().cpu() if self.prediction_segmentations is not None else None),
            VisualisationData(
                "line", "Loss", "Loss function",
                self.loss.detach().cpu() if self.loss is not None else None,
                text=self.loss_names[0]),
        ]
        return data
