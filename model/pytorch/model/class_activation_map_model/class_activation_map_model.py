"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from model.pytorch.base_model import BasePytorchModel
from model.pytorch.model.class_activation_map_model.class_activation_map_network import ClassActivationMapNetwork
from model.builder import MODELS


@MODELS.register_module(name='CLASS_ACTIVATION_MAP_MODEL')
class ClassActivationMapModel(BasePytorchModel):

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
        self.network = ClassActivationMapNetwork()
        self.network.build()

