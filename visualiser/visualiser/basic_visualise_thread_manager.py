"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports
from abc import ABC
from queue import Queue

# Internal imports


class BasicVisualiseThreadManager(ABC):
    """
    Simple thread manager which receives information and sends it to the thread
    """

    def __init__(self):
        self.visualiser = None

    def build_and_start(self, visualiser_class):
        """

        Args:
            visualiser_class: a visualiser, either tensorboard or visdom based

        Returns:

        """
        queue = Queue()
        self.visualiser = visualiser_class(queue)
        self.visualiser.start()

    def visualise(self, iteration, key, data, **kwargs):
        """
        Sends data to the visualiser thread
        Args:
            iteration: integer representing the order of the visualisation
            key: the key in RUNNER.VISUALSATIONS
            data: data to send to the visualiser
        """
        self.visualiser.queue.put((iteration, key, data, kwargs))
