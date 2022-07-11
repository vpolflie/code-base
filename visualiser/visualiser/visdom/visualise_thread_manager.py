"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports

# Internal imports
from visualiser.visdom.basic_visualiser import BasicVisualiser
from visualiser.basic_visualise_thread_manager import BasicVisualiseThreadManager
from visualiser.builder import VISUALIZERS


@VISUALIZERS.register_module(name="VISDOM")
class VisdomVisualiseThreadManager(BasicVisualiseThreadManager):
    """
    Simple thread manager which receives information and sends it to the thread
    """

    def __init__(self):
        super().__init__()
        self.build_and_start(BasicVisualiser)
