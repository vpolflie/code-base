"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from torch.distributed import init_process_group

# Internal imports
from model.pytorch.base_model import BasePytorchModel
from .hrnet_network import HRNetNetwork
from model.builder import MODELS


@MODELS.register_module(name='HRNET_MODEL')
class HRNetModel(BasePytorchModel):

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
        self.network = HRNetNetwork()
        self.network.build()
