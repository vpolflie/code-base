"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch.nn as nn
import torch.nn.functional as F

# Internal imports
import config.config as cfg
from model.pytorch.base_block import BaseBlock
from model.pytorch.core.backbone.resnet_backbone import ResNetBasicBlock, ResNetBottleneck


class ClassActivationMapBlock(BaseBlock):

    """
        Example Network Block used to define separate parts of the network
    """

    def __init__(self):
        """
        Initialise the variables in the block
        """
        super().__init__()

        # Variable used to save the model
        self.save_name = "class_activation_map_block"

        # Block variables
        self.entry_block = None
        self.resnet_layers = None
        self.classifier = None
        self.dense = None
        self.kernel_size = [3, 5, 7, 3]
        self.dilation = [1, 1, 1, 1]
        self.padding_numbers = [1, 2, 3, 1]
        self.strides = [(2, 2), (2, 2), (2, 1), (1, 1)]

    def build(self):
        """
        Build the actual network graph
        """

        # Create entry layers
        entry_block_list = [
            nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, cfg.MODEL.BASE_CONVOLUTIONAL_FILTERS,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cfg.MODEL.BASE_CONVOLUTIONAL_FILTERS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        # Create resnet encoder
        resnet_layers_list = []
        current_output_filters = cfg.MODEL.BASE_CONVOLUTIONAL_FILTERS
        for i in range(cfg.MODEL.LAYERS):
            current_input_filters = cfg.MODEL.BASE_CONVOLUTIONAL_FILTERS * (2 ** i)
            current_output_filters = cfg.MODEL.BASE_CONVOLUTIONAL_FILTERS * (2 ** (i + 1))

            resnet_layers_list.append(ResNetBasicBlock(
                current_input_filters,
                current_output_filters,
                stride=self.strides[i],
                dilation=self.dilation[i],
            ))

        resnet_layers_list.append(ResNetBottleneck(
           current_output_filters,
           current_output_filters,
           stride=1,
           dilation=self.dilation[-1],
        ))

        self.entry_block = nn.Sequential(*entry_block_list)
        self.resnet_layers = nn.Sequential(*resnet_layers_list)

        self.classifier = nn.Conv2d(current_output_filters, cfg.MODEL.OUTPUT_CHANNELS, 1, bias=False)
        self.dense = nn.Linear(cfg.MODEL.OUTPUT_CHANNELS, cfg.MODEL.OUTPUT_CHANNELS, bias=False)

    def forward(self, x):
        """
        Define the forward function that will feed data through the network graph
        """
        # Feed through encoder
        embedding = self.entry_block(x)
        embedding = self.resnet_layers(embedding)

        # Global pool final layer
        #avg_global_pooling = torch.mean(embedding, dim=(2, 3), keepdim=True)

        # Get predictions
        predictions = self.dense(self.classifier(embedding).view(embedding.shape[0], -1, cfg.MODEL.OUTPUT_CHANNELS))
        class_activation_map = F.relu(F.conv2d(embedding, self.classifier.weight))
        return predictions, class_activation_map
