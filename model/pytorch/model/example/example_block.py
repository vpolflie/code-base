"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch.nn as nn

# Internal imports
import config.config as cfg
from model.pytorch.base_block import BaseBlock


class ExampleBlock(BaseBlock):

    """
        Example Network Block used to define separate parts of the network
    """

    def __init__(self):
        """
        Initialise the variables in the block
        """
        super().__init__()

        # Variable used to save the model
        self.save_name = "example_block"

        # Block variables
        self.encoder = None
        self.decoder = None

    def build(self):
        """
        Build the actual network graph
        """
        # Encoder Network
        encoder_list = [
            nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, cfg.MODEL.FILTERS * (2 ** 0), 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.MODEL.FILTERS * (2 ** 0), cfg.MODEL.FILTERS * (2 ** 1), 3, stride=2),
            nn.ReLU(),
        ]
        # Decoder Network
        decoder_list = [
            nn.ConvTranspose2d(cfg.MODEL.FILTERS * (2 ** 1), cfg.MODEL.FILTERS * (2 ** 0), 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(cfg.MODEL.FILTERS * (2 ** 0), cfg.MODEL.OUTPUT_CHANNELS, 4, 2),
            nn.Softmax(dim=1)
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        """
        Define the forward function that will feed data through the network graph
        """
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        return output
