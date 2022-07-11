"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from torch.distributed import init_process_group

# Internal imports
from model.pytorch.base_model import BasePytorchModel
from .deeplab_v3p_network import DeeplabV3PNetwork
from model.builder import MODELS


@MODELS.register_module(name='DEEPLABV3P_MODEL')
class DeeplabV3PModel(BasePytorchModel):

    """
        Base Pytorch Model Class that needs to be implemented.
        This class is mainly a pytorch interface between the framework and the python api structure
    """

    def __init__(self):
        """
            Example model initialisation.
        """
        super().__init__()

        self.network = None
        self.network_caller = None

    def build(self):
        self.network = DeeplabV3PNetwork()
        self.network.build()
