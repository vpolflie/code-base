"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

# Internal imports
from model.pytorch.base_block import BaseBlock
from model.pytorch.core.general.aspp import ASPP

import config.config as cfg


class DeeplabV3PlusBlock(BaseBlock):

    """
        Example Network Block used to define separate parts of the network
    """

    def __init__(self, name="semi_supervised_segmentation_block"):
        """
        Initialise the variables in the block
        """
        super().__init__(name=name)

        # Block parameters
        self.aspp_dilate = [6, 12, 18]
        self.in_channels = 2048
        self.low_level_channels = 256

        # Block variables
        self.project = None
        self.aspp = None
        self.classifier = None

    def build(self):
        """
        Build the actual network graph
        """
        self.project = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, (1, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(self.in_channels, self.aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.MODEL.OUTPUT_CHANNELS, (1, 1))
        )
        self._init_weight()

    def forward(self, x):
        """
        Define the forward function that will feed data through the network graph
        """
        low_level_feature = self.project(x['low_level'])
        output_feature = self.aspp(x['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
