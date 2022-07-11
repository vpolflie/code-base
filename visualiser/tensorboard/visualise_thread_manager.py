"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from visualiser.tensorboard.basic_visualiser import BasicVisualiser
from visualiser.basic_visualise_thread_manager import BasicVisualiseThreadManager
from visualiser.builder import VISUALIZERS


@VISUALIZERS.register_module(name="TENSORBOARD")
class TensorboardVisualiseThreadManager(BasicVisualiseThreadManager):
    """
    Simple thread manager which receives information and sends it to the thread
    """

    def __init__(self):
        super().__init__()
        self.build_and_start(BasicVisualiser)
