"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from model.pytorch.base_model import BasePytorchModel
from model.pytorch.model.example.example_network import ExampleNetwork
from model.builder import MODELS


@MODELS.register_module(name='PYTORCH_EXAMPLE_MODEL')
class PytorchExampleModel(BasePytorchModel):

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
        self.network = ExampleNetwork()
        self.network.build()

