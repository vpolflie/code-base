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

from .backbone_block import BackboneBlock
from .deeplab_v3_plus_block import DeeplabV3PlusBlock


class SemiSupervisedDeeplabV3PNetwork(BasePytorchNetwork):

    """
        Example Network Class.
    """

    def __init__(self):
        """
            Base Model initialisation.
        """
        super().__init__()

        # Setup Logger Info
        self.loss_names = ["Cross Entropy Loss 1", "Pseudo Loss 1",
                           "Cross Entropy Loss 2", "Pseudo Loss 2",
                           "Recall 1", "Precision 1",
                           "Recall 2", "Precision 2", ]

        self.max_length = ["%0.7f", "%0.7f", "%0.7f", "%0.7f", "%0.7f", "%0.7f", "%0.7f", "%0.7f"]
        self.comparison = ["<", "<", "<", "<", ">", ">", ">", ">"]

        # Block/optimizers pairs for save/load
        self.network_blocks = []
        self.optimizers = []

        # Network Variables
        self.semantic_segmentation_block_1 = None
        self.semantic_segmentation_block_2 = None
        self.backbone_1 = None
        self.backbone_2 = None
        self.optimizer_1 = None
        self.optimizer_2 = None
        self.scheduler_1 = None
        self.scheduler_2 = None

        self.images = None
        self.segmentations = None
        self.roads = None
        self.prediction_segmentations_1 = None
        self.prediction_segmentations_2 = None

        # losses
        self.cross_entropy_loss_1 = None
        self.cross_entropy_loss_2 = None
        self.pseudo_loss_1 = None
        self.pseudo_loss_2 = None
        self.loss_1 = None
        self.loss_2 = None

        self.pseudo_weight = cfg.MODEL.PSEUDO_WEIGHT

        #self.cross_entropy_loss_function = nn.BCELoss() if cfg.MODEL.OUTPUT_CHANNELS == 1 \
        #    else nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.cross_entropy_loss_function = FocalLoss(ignore_index=0)

        # Metrics
        self.precision_function = Precision()
        self.recall_function = Recall()
        self.recall_1 = None
        self.recall_2 = None
        self.precision_1 = None
        self.precision_2 = None

        # Extra visualisation variables
        self.probability_visualisation_1 = None
        self.probability_visualisation_2 = None

    def build(self):
        """
            Build the entire network using blocks and returns a dictionary of optimizers
        """
        self.backbone_1 = BackboneBlock(name="backbone_1")
        self.backbone_1.build()
        self.semantic_segmentation_block_1 = DeeplabV3PlusBlock(name="deeplab_v3_plus_block_1")
        self.semantic_segmentation_block_1.build()
        self.optimizer_1 = torch.optim.SGD(params=[
            {'params': self.backbone_1.parameters(), 'lr': 0.1*cfg.MODEL.LEARNING_RATE},
            {'params': self.semantic_segmentation_block_1.parameters(), 'lr': cfg.MODEL.LEARNING_RATE},
        ], lr=cfg.MODEL.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

        self.network_blocks.append(self.backbone_1)
        self.network_blocks.append(self.semantic_segmentation_block_1)

        self.scheduler_1 = LearningRateScheduler(
            self.optimizer_1,
            PolyLR,
            warm_up=cfg.MODEL.WARM_UP if hasattr(cfg.MODEL, "WARM_UP") else False
        )

        self.optimizers.append(("optimizer_1", self.optimizer_1, self.scheduler_1))

        self.backbone_2 = BackboneBlock(name="backbone_2")
        self.backbone_2.build()
        self.semantic_segmentation_block_2 = DeeplabV3PlusBlock(name="deeplab_v3_plus_block_2")
        self.semantic_segmentation_block_2.build()
        self.optimizer_2 = torch.optim.SGD(params=[
            {'params': self.backbone_2.parameters(), 'lr': 0.1 * cfg.MODEL.LEARNING_RATE},
            {'params': self.semantic_segmentation_block_2.parameters(), 'lr': cfg.MODEL.LEARNING_RATE},
        ], lr=cfg.MODEL.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

        self.network_blocks.append(self.backbone_2)
        self.network_blocks.append(self.semantic_segmentation_block_2)

        self.scheduler_2 = LearningRateScheduler(
            self.optimizer_2,
            PolyLR,
            warm_up=cfg.MODEL.WARM_UP if hasattr(cfg.MODEL, "WARM_UP") else False
        )

        self.optimizers.append(("optimizer_2", self.optimizer_2, self.scheduler_2))

    def set_input(self, x):
        """
            Set the input variables for the networks
            Args:
                x: input parameters
        """
        self.images = x[0].to(self.device) if x[0] is not None else None
        self.segmentations = x[1].type(torch.LongTensor).to(self.device) if x[1] is not None else None
        self.roads = x[2].type(torch.LongTensor).to(self.device) if x[2] is not None else None

    def feed_forward(self):
        """
            Forward pass through the network
        """
        shape = self.images.shape[2:]
        features_1 = self.backbone_1(self.images)
        prediction_1 = self.semantic_segmentation_block_1(features_1)
        self.prediction_segmentations_1 = F.interpolate(prediction_1, size=shape, mode='bilinear', align_corners=False)

        features_2 = self.backbone_2(self.images)
        prediction_2 = self.semantic_segmentation_block_2(features_2)
        self.prediction_segmentations_2 = F.interpolate(prediction_2, size=shape, mode='bilinear', align_corners=False)

    def calculate_losses(self):
        """
            Calculate all the loss terms of the network
        """

        self.cross_entropy_loss_1 = self.cross_entropy_loss_function(
            self.prediction_segmentations_1,
            self.segmentations.squeeze()
        )

        self.cross_entropy_loss_2 = self.cross_entropy_loss_function(
            self.prediction_segmentations_2,
            self.segmentations.squeeze(1)
        )

        self.pseudo_loss_1 = self.cross_entropy_loss_function(
            self.prediction_segmentations_1,
            torch.argmax(self.prediction_segmentations_2.detach(), dim=1)
        )

        self.pseudo_loss_2 = self.cross_entropy_loss_function(
            self.prediction_segmentations_2,
            torch.argmax(self.prediction_segmentations_1.detach(), dim=1)
        )

        probability_predictions_1 = torch.amax(torch.softmax(self.prediction_segmentations_1, dim=1), dim=1, keepdim=True)
        probability_predictions_2 = torch.amax(torch.softmax(self.prediction_segmentations_1, dim=1), dim=1, keepdim=True)
        probability_ground_truth = torch.gather(torch.softmax(self.prediction_segmentations_1, dim=1), 1, self.segmentations)
        self.probability_visualisation_1 = torch.abs(probability_predictions_1 - probability_ground_truth)
        self.probability_visualisation_2 = torch.abs(probability_predictions_2 - probability_ground_truth)

        self.loss_1 = torch.mean(self.cross_entropy_loss_1 * (1 - self.roads[:, 1:2]) +
                                 self.pseudo_weight * self.pseudo_loss_1 * self.roads[:, 1:2])
        self.loss_2 = torch.mean(self.cross_entropy_loss_2 * (1 - self.roads[:, 1:2]) +
                                 self.pseudo_weight * self.pseudo_loss_2 * self.roads[:, 1:2])

        self.cross_entropy_loss_1 = \
            torch.mean(self.cross_entropy_loss_1 * (1 - self.roads[:, 1:2]))
        self.cross_entropy_loss_2 = \
            torch.mean(self.cross_entropy_loss_2 * (1 - self.roads[:, 1:2]))
        self.pseudo_loss_1 = torch.mean(self.pseudo_loss_1 * self.roads[:, 1:2])
        self.pseudo_loss_2 = torch.mean(self.pseudo_loss_2 * self.roads[:, 1:2])

        self.precision_1 = self.precision_function(
            torch.swapaxes(torch.nn.functional.one_hot(self.segmentations.detach(), cfg.MODEL.OUTPUT_CHANNELS), 4, 0),
            torch.swapaxes(torch.nn.functional.one_hot(torch.argmax(self.prediction_segmentations_1.detach(), dim=1,
                                                                    keepdim=True), cfg.MODEL.OUTPUT_CHANNELS), 4, 0))
        self.precision_2 = self.precision_function(
            torch.swapaxes(torch.nn.functional.one_hot(self.segmentations.detach(), cfg.MODEL.OUTPUT_CHANNELS), 4, 0),
            torch.swapaxes(torch.nn.functional.one_hot(torch.argmax(self.prediction_segmentations_2.detach(), dim=1,
                                                                    keepdim=True), cfg.MODEL.OUTPUT_CHANNELS), 4, 0))
        self.recall_1 = self.recall_function(
            torch.swapaxes(torch.nn.functional.one_hot(self.segmentations.detach(), cfg.MODEL.OUTPUT_CHANNELS), 4, 0),
            torch.swapaxes(torch.nn.functional.one_hot(torch.argmax(self.prediction_segmentations_1.detach(), dim=1,
                                                                    keepdim=True), cfg.MODEL.OUTPUT_CHANNELS), 4, 0))
        self.recall_2 = self.recall_function(
            torch.swapaxes(torch.nn.functional.one_hot(self.segmentations.detach(), cfg.MODEL.OUTPUT_CHANNELS), 4, 0),
            torch.swapaxes(torch.nn.functional.one_hot(torch.argmax(self.prediction_segmentations_2.detach(), dim=1,
                                                                    keepdim=True), cfg.MODEL.OUTPUT_CHANNELS), 4, 0))

    def optimize_step(self, losses):
        """
            Optimize the networks over the corresponding losses
            Args:
                losses: list of losses to optimize over
        """

        # Calculate losses
        self.loss_1.backward()
        self.optimizer_1.step()
        self.optimizer_1.zero_grad()

        self.loss_2.backward()
        self.optimizer_2.step()
        self.optimizer_2.zero_grad()

    def set_output(self):
        """
            Set the output variables for the networks
            Returns: two tuples of output variables, first one being the results and the second on being the loss
        """
        return (torch.softmax((self.prediction_segmentations_1 + self.prediction_segmentations_2) / 2, dim=1),), self.get_losses()

    def update_learning_rate(self, update_percentage):
        """
        Update the learning rate of the optimizers

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.scheduler_1.step(update_percentage)
        self.scheduler_2.step(update_percentage)

    def update_optimizer_warm_up(self, update_percentage):
        """
        Update the learning rate of the optimizers in terms of warm up. This means slowly building up the learning
        rate from 0 to the required value so the optimizer can figure out it's gradients

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.scheduler_1.step_warm_up(update_percentage)
        self.scheduler_2.step_warm_up(update_percentage)

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
        data = []
        data += [VisualisationData("images", "Image", "Input Images",
                                   self.images[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.images is not None else []
        data += [VisualisationData("segmentations", "Segmentation", "Ground truth Segmentations",
                                   torch.swapaxes(torch.nn.functional.one_hot(
                                       self.segmentations[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu(), cfg.MODEL.OUTPUT_CHANNELS)
                                       , 1, 4).squeeze())] \
                                   if self.segmentations is not None else []
        data += [VisualisationData("segmentations", "Predictions 1", "Predicted Segmentations 1",
                                   self.prediction_segmentations_1[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.prediction_segmentations_1 is not None else []
        data += [VisualisationData("heatmap", "Difference Heatmap 1", "Difference Heatmap 1",
                                   self.probability_visualisation_1[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu(), xmin=0, xmax=1)] \
                                   if self.probability_visualisation_1 is not None else []
        data += [VisualisationData("segmentations", "Predictions 2", "Predicted Segmentations 2",
                                   self.prediction_segmentations_2[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.prediction_segmentations_2 is not None else []
        data += [VisualisationData("heatmap", "Difference Heatmap 2", "Difference Heatmap 2",
                                   self.probability_visualisation_2[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu(), xmin=0, xmax=1)] \
                                   if self.probability_visualisation_2 is not None else []
        data += [VisualisationData("segmentations", "Roads", "Roads",
                                   self.roads[-1 * cfg.VISUALISER.MAX_ELEMENTS:].detach().cpu())] \
                                   if self.roads is not None else []

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
            torch.mean(self.cross_entropy_loss_1).detach().cpu() if self.cross_entropy_loss_1 is not None else None,
            torch.mean(self.pseudo_loss_1).detach().cpu() if self.pseudo_loss_1 is not None else None,
            torch.mean(self.cross_entropy_loss_2).detach().cpu() if self.cross_entropy_loss_2 is not None else None,
            torch.mean(self.pseudo_loss_2).detach().cpu() if self.pseudo_loss_2 is not None else None,
            self.recall_1.detach().cpu() if self.recall_1 is not None else None,
            self.precision_1.detach().cpu() if self.precision_1 is not None else None,
            self.recall_2.detach().cpu() if self.recall_2 is not None else None,
            self.precision_2.detach().cpu() if self.precision_2 is not None else None,
        )
