"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch
import torch.nn.functional as F

# Internal imports
import config.config as cfg
from model.pytorch.base_network import BasePytorchNetwork
from model.pytorch.model.class_activation_map_model.class_activation_map_block import ClassActivationMapBlock
from visualiser.visualisation_data import VisualisationData


class ClassActivationMapNetwork(BasePytorchNetwork):

    """
        Example Network Class.
    """

    def __init__(self):
        """
            Base Model initialisation.
        """
        super().__init__()

        # Setup Logger Info
        self.loss_names = ["Multi Label Soft Margin Loss", "Accuracy"]
        self.max_length = ["%0.7f", "%0.7f"]
        self.comparison = ["<", ">"]

        # Block/optimizers pairs for save/load
        self.network_blocks = []

        # Network Variables
        self.class_activation_map_block = None
        self.class_activation_map_optimizer = None

        self.images = None
        self.classes = None
        self.predictions = None
        self.class_activation_map = None

        self.loss = None
        self.accuracy = None
        self.accuracy_criterion = lambda x, y: torch.sum(torch.eq(x, y)) / x.shape[0]
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def build(self):
        """
            Build the entire network using blocks and returns a dictionary of optimizers
        """
        self.class_activation_map_block = ClassActivationMapBlock()
        self.class_activation_map_block.build()
        self.class_activation_map_optimizer = torch.optim.Adam(self.class_activation_map_block.parameters(),
                                                               lr=cfg.MODEL.LEARNING_RATE)
        self.network_blocks.append((self.class_activation_map_block, self.class_activation_map_optimizer))

    def set_input(self, x):
        """
            Set the input variables for the networks
            Args:
                x: input parameters
        """
        self.images = x[0].to(self.device)
        self.classes = x[1].to(self.device)

    def feed_forward(self):
        """
            Forward pass through the network
        """
        self.predictions, self.class_activation_map = self.class_activation_map_block(self.images)
        self.classes = self.classes.view(-1, 1, cfg.MODEL.OUTPUT_CHANNELS).tile((1, self.predictions.shape[1], 1))
        self.classes = self.classes.view(-1, cfg.MODEL.OUTPUT_CHANNELS)
        self.predictions = self.predictions.view(-1, cfg.MODEL.OUTPUT_CHANNELS)
        self.class_activation_map = torch.sum(self.class_activation_map, dim=1, keepdim=True)
        min_activation, _ = torch.min(self.class_activation_map, dim=0, keepdim=True)
        max_activation, _ = torch.max(self.class_activation_map, dim=0, keepdim=True)
        self.class_activation_map = (self.class_activation_map - min_activation) / (max_activation - min_activation)

    def calculate_losses(self):
        """
            Calculate all the loss terms of the network
        """
        self.loss = self.loss_criterion(torch.softmax(self.predictions, dim=1),
                                        self.classes.argmax(dim=-1))
        self.accuracy = self.accuracy_criterion(self.predictions.argmax(dim=-1),
                                                self.classes.argmax(dim=-1))

    def optimize_step(self, losses):
        """
            Optimize the networks over the corresponding losses
            Args:
                losses: list of losses to optimize over
        """
        self.loss.backward()
        self.class_activation_map_optimizer.step()
    
    def set_output(self):
        """
            Set the output variables for the networks
            Returns: two tuples of output variables, first one being the results and the second on being the loss
        """
        return (self.predictions, self.class_activation_map), (self.loss, self.accuracy)

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
                self.images[:-1 * cfg.VISUALISER.MAX_ELEMENTS].detach().cpu() if self.images is not None else None),
            VisualisationData(
                "images", "CAM", "CAM",
                self.class_activation_map[:-1 * cfg.VISUALISER.MAX_ELEMENTS].detach().cpu() if self.class_activation_map is not None else None),
            VisualisationData(
                "line", "Loss", "Loss function",
                self.loss.detach().cpu() if self.loss is not None else None,
                text=self.loss_names[0]),
            VisualisationData(
                "line", "Accuracy", "Accuracy",
                self.accuracy.detach().cpu() if self.loss is not None else None,
                text=self.loss_names[1]),
        ]
        return data
