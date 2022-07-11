"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal imports
import config.config as cfg
from model.pytorch.base_network import BasePytorchNetwork
from model.pytorch.core.loss.focal import FocalLoss
from model.pytorch.core.loss.precision import Precision
from model.pytorch.core.loss.recall import Recall
from model.pytorch.core.scheduler.learning_rate_scheduler import LearningRateScheduler
from model.pytorch.core.scheduler.poly_scheduler import PolyLR
from visualiser.visualisation_data import VisualisationData

from .hrnet_block import HRNetBlock


class HRNetNetwork(BasePytorchNetwork):

    """
        Example Network Class.
    """

    def __init__(self):
        """
            Base Model initialisation.
        """
        super().__init__()

        # Setup Logger Info
        self.loss_names = ["Cross Entropy Loss 1", "Recall 1", "Precision 1"]

        self.max_length = ["%0.7f", "%0.7f", "%0.7f"]
        self.comparison = ["<", ">", ">"]

        # Block/optimizers pairs for save/load
        self.network_blocks = []
        self.optimizers = []

        # Network Variables
        self.semantic_segmentation_block = None
        self.optimizer = None
        self.scheduler = None

        self.images = None
        self.segmentations = None
        self.prediction_segmentations = None

        # losses
        self.cross_entropy_loss = None
        self.loss = None

        self.cross_entropy_loss_function = nn.BCELoss() if cfg.MODEL.OUTPUT_CHANNELS == 1 else FocalLoss()

        # Metrics
        self.precision_function = Precision(ignore_index=0)
        self.recall_function = Recall(ignore_index=0)
        self.recall = None
        self.precision = None

    def build(self):
        """
            Build the entire network using blocks and returns a dictionary of optimizers
        """
        self.semantic_segmentation_block = HRNetBlock(name="hrnet_block")
        self.semantic_segmentation_block.build()
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.semantic_segmentation_block.parameters(), 'lr': cfg.MODEL.LEARNING_RATE},
        ], lr=cfg.MODEL.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

        self.network_blocks.append(self.semantic_segmentation_block)

        self.scheduler = LearningRateScheduler(
            self.optimizer,
            PolyLR,
            warm_up=cfg.MODEL.WARM_UP if hasattr(cfg.MODEL, "WARM_UP") else False
        )

        self.optimizers.append(("optimizer", self.optimizer, self.scheduler))

    def set_input(self, x):
        """
            Set the input variables for the networks
            Args:
                x: input parameters
        """
        self.images = x[0].to(self.device) if x[0] is not None else None
        self.segmentations = x[1].type(torch.LongTensor).to(self.device) if x[1] is not None else None

    def feed_forward(self):
        """
            Forward pass through the network
        """
        shape = self.images.shape[2:]
        prediction = self.semantic_segmentation_block(self.images)
        self.prediction_segmentations = F.interpolate(prediction, size=shape, mode='bilinear', align_corners=False)

    def calculate_losses(self):
        """
            Calculate all the loss terms of the network
        """

        self.cross_entropy_loss = self.cross_entropy_loss_function(
            self.prediction_segmentations,
            self.segmentations.squeeze(1)
        )

        self.loss = torch.mean(self.cross_entropy_loss)

        self.precision = self.precision_function(
            torch.transpose(torch.reshape(torch.nn.functional.one_hot(
                self.segmentations.detach().squeeze(), cfg.MODEL.OUTPUT_CHANNELS), (-1, cfg.MODEL.OUTPUT_CHANNELS)), 0, 1),
            torch.transpose(torch.reshape(torch.nn.functional.one_hot(
                torch.argmax(self.prediction_segmentations.detach(), dim=1, keepdim=True).squeeze(),
                cfg.MODEL.OUTPUT_CHANNELS), (-1, cfg.MODEL.OUTPUT_CHANNELS)), 0, 1))
        self.recall = self.recall_function(
            torch.transpose(torch.reshape(torch.nn.functional.one_hot(
                self.segmentations.detach().squeeze(), cfg.MODEL.OUTPUT_CHANNELS), (-1, cfg.MODEL.OUTPUT_CHANNELS)), 0, 1),
            torch.transpose(torch.reshape(torch.nn.functional.one_hot(
                torch.argmax(self.prediction_segmentations.detach(), dim=1, keepdim=True).squeeze(),
                cfg.MODEL.OUTPUT_CHANNELS), (-1, cfg.MODEL.OUTPUT_CHANNELS)), 0, 1))

    def optimize_step(self, losses):
        """
            Optimize the networks over the corresponding losses
            Args:
                losses: list of losses to optimize over
        """

        # Calculate losses
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_output(self):
        """
            Set the output variables for the networks
            Returns: two tuples of output variables, first one being the results and the second on being the loss
        """
        return (torch.softmax(self.prediction_segmentations, dim=1),), self.get_losses()

    def update_learning_rate(self, update_percentage):
        """
        Update the learning rate of the optimizers

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.scheduler.step(update_percentage)

    def update_optimizer_warm_up(self, update_percentage):
        """
        Update the learning rate of the optimizers in terms of warm up. This means slowly building up the learning
        rate from 0 to the required value so the optimizer can figure out it's gradients

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.scheduler.step_warm_up(update_percentage)

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
        probability_visualisation = None
        if self.segmentations is not None:
            probability_predictions = torch.amax(torch.softmax(self.prediction_segmentations, dim=1), dim=1, keepdim=True)
            probability_ground_truth = torch.gather(torch.softmax(self.prediction_segmentations, dim=1), 1, self.segmentations)
            probability_visualisation = torch.abs(probability_predictions - probability_ground_truth)

        data = []
        data += [VisualisationData("images", "Image", "Input Images",
                                   self.images[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.images is not None else []
        data += [VisualisationData("segmentations", "Segmentation", "Ground truth Segmentations",
                                   torch.swapaxes(torch.nn.functional.one_hot(
                                       self.segmentations[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu(), cfg.MODEL.OUTPUT_CHANNELS)
                                       , 1, 4).squeeze())] \
                                   if self.segmentations is not None else []
        data += [VisualisationData("segmentations", "Predictions", "Predicted Segmentations",
                                   self.prediction_segmentations[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.prediction_segmentations is not None else []
        data += [VisualisationData("heatmap", "Difference Heatmap", "Difference Heatmap",
                                   probability_visualisation[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu(), xmin=0, xmax=1)] \
                                   if probability_visualisation is not None else []

        losses = self.get_losses()
        for index in range(len(losses)):
            data.append(
                VisualisationData("line", self.loss_names[index], "Loss function %s" % self.loss_names[index],
                                  losses[index],
                                  text=self.loss_names[index])
            )
        return data

    def get_losses(self):
        """
        Helper function to return all the losses
        Returns:
            A list of losses
        """
        return (
            torch.mean(self.cross_entropy_loss).detach().cpu() if self.cross_entropy_loss is not None else None,
            self.recall.detach().cpu() if self.recall is not None else None,
            self.precision.detach().cpu() if self.precision is not None else None,
        )
