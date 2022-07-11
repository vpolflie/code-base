"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import os
import threading
from torch.utils.tensorboard import SummaryWriter

# Internal imports
import config.config as cfg
from tools.data.shape import array_to_image, array_to_segmentation_image, array_to_grid
from tools.data.type import convert_to_numpy
from tools.path_utils import create_folders


class BasicVisualiser(threading.Thread):

    def __init__(self, queue):
        """
        A tensorboard visualiser

        Args:
            queue: a threading queue
        """
        super(BasicVisualiser, self).__init__()

        # Tensorboard variables
        self.tensorboard_writers = {}
        for visualiser in cfg.RUNNER.VISUALISATIONS:
            path = os.path.join("meta_data", "tensorboard", cfg.RUNNER.OUTPUT, visualiser)
            create_folders(path, remove=True)
            self.tensorboard_writers[visualiser] = SummaryWriter(path)
        self.queue = queue
        self.daemon = True

        # Visualiser variables
        self.image_plotter = None
        self.images_plotter = None
        self.line_plotter = None
        self.histogram_plotter = None

        # Initialiser
        self.initialise_visualisations()

    def run(self):
        """
        Threading runner, gets information and visualises it
        """
        while True:
            data = self.queue.get()
            self.visualise(*data)

    def initialise_visualisations(self):
        """
            Initialise all the visualiser visualisers
        """
        pass

    def visualise(self, iteration, visualisation_index, visualisation_data, **kwargs):
        """
        Visualise data
        Args:
            visualisation_index: index of the visualisations
            iteration: current iteration
            visualisation_data: data to visualise
        """
        environment = cfg.RUNNER.VISUALISATIONS[visualisation_index]

        for data in visualisation_data:
            if data.data is not None:
                if data.tag == "line":
                    self.tensorboard_writers[environment].add_scalar(data.text,
                                                                     convert_to_numpy([data.data]),
                                                                     iteration)
                if data.tag == "histogram":
                    self.tensorboard_writers[environment].add_histogram(data.plot_tag,
                                                                        convert_to_numpy(data.data),
                                                                        iteration)
                if data.tag == "image":
                    self.tensorboard_writers[environment].add_image(data.plot_tag,
                                                                    array_to_image(convert_to_numpy(data.data))[0],
                                                                    iteration)
                if data.tag == "images":
                    images = \
                        array_to_grid(array_to_image(convert_to_numpy(data.data)), images_per_row=data.images_per_row)
                    self.tensorboard_writers[environment].add_image(data.plot_tag,
                                                                    images,
                                                                    iteration)
                if data.tag == "segmentations":
                    images = \
                        array_to_grid(array_to_image(convert_to_numpy(data.data)), images_per_row=data.images_per_row)
                    self.tensorboard_writers[environment].add_image(data.plot_tag,
                                                                    images,
                                                                    iteration)
