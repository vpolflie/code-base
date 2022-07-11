"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC, abstractmethod
import torch.nn as nn

# Internal imports


class BaseBlock(nn.Module, ABC):

    """
        Basic Network Block used to define separate parts of the network
    """

    def __init__(self, name="block"):
        """
        Initialise the variables in the block
        """
        super().__init__()

        # Variable used to save the model
        self.save_name = name

    @abstractmethod
    def build(self):
        """
        Build the actual network graph
        """
        pass
