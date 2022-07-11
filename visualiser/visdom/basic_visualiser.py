"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import threading

import numpy as np
from visdom import Visdom

# Internal imports
import config.config as cfg
from tools.data.shape import array_to_image, array_to_grid, array_to_segmentation_image
from tools.data.type import convert_to_numpy
from tools.data.value import denormalize
from visualiser.visdom.util import *


class BasicVisualiser(threading.Thread):

    def __init__(self, queue):
        """
        A visdom visualiser

        Args:
            queue: a threading queue
        """
        super(BasicVisualiser, self).__init__()

        # Visdom variables
        self.visdom = Visdom()
        for visualisation in cfg.RUNNER.VISUALISATIONS:
            self.visdom.delete_env(env=cfg.RUNNER.OUTPUT + "_" + visualisation)
        self.queue = queue
        self.daemon = True

        # Visualiser variables
        self.image_plotter = None
        self.images_plotter = None
        self.bounding_box_images_plotter = None
        self.bounding_box_heatmap_images_plotter = None
        self.bounding_box_iou_images_plotter = None
        self.line_plotter = None
        self.histogram_plotter = None
        self.heatmap_plotter = None
        self.text_plotter = None

        # Initialiser
        self.initialise_visualisations()

    def run(self):
        """
        Threading runner, gets information and visualises it
        """
        while True:
            data = self.queue.get()
            self.visualise(*data[0:3], **data[3])

    def initialise_visualisations(self):
        """
            Initialise all the visualiser visualisers
        """
        self.text_plotter = TextPlotter(self.visdom)
        self.image_plotter = ImagePlotter(self.visdom)
        self.images_plotter = ImagesPlotter(self.visdom)
        self.line_plotter = LinePlotter(self.visdom)
        self.histogram_plotter = HistogramPlotter(self.visdom)
        self.heatmap_plotter = HeatmapPlotter(self.visdom)

    def visualise(self, iteration, visualisation_index, visualisation_data, **kwargs):
        """
        Visualise data
        Args:
            visualisation_index: index of the visualisations
            iteration: current iteration
            visualisation_data: list of data to visualise
        """
        environment_name = cfg.RUNNER.OUTPUT + "_" + cfg.RUNNER.VISUALISATIONS[visualisation_index]
        if visualisation_data is not None:
            for data in visualisation_data:
                if data.data is not None:
                    if data.tag == "line":
                        self.line_plotter.plot(environment_name,
                                               data.plot_tag,
                                               data.title,
                                               iteration,
                                               convert_to_numpy(data.data),
                                               data.text,
                                               data.moving_average_length)
                    if data.tag == "histogram":
                        self.histogram_plotter.plot(environment_name,
                                                    data.plot_tag,
                                                    data.title,
                                                    convert_to_numpy(data.data),
                                                    data.number_of_bins)
                    if data.tag == "image":
                        if "normalize" in kwargs.keys() and kwargs["normalize"]:
                            image = denormalize(convert_to_numpy(data.data), **kwargs)
                        else:
                            image = convert_to_numpy(data.data)

                        self.image_plotter.plot(environment_name,
                                                data.plot_tag,
                                                data.title,
                                                array_to_grid(array_to_image(image, **kwargs))[0],
                                                data.store_history)
                    if data.tag == "images":
                        if "normalize" in kwargs.keys() and kwargs["normalize"]:
                            image = denormalize(convert_to_numpy(data.data), **kwargs)
                        else:
                            image = convert_to_numpy(data.data)

                        self.images_plotter.plot(environment_name,
                                                 data.plot_tag,
                                                 data.title,
                                                 array_to_image(image),
                                                 data.images_per_row)
                    if data.tag == "segmentation":
                        self.image_plotter.plot(environment_name,
                                                data.plot_tag,
                                                data.title,
                                                array_to_grid(array_to_segmentation_image(
                                                    convert_to_numpy(data.data),
                                                    **kwargs))[0], data.store_history)
                    if data.tag == "segmentations":
                        self.images_plotter.plot(environment_name,
                                                 data.plot_tag,
                                                 data.title,
                                                 array_to_segmentation_image(convert_to_numpy(data.data), **kwargs),
                                                 data.images_per_row)
                    if data.tag == "heatmap":
                        self.heatmap_plotter.plot(environment_name,
                                                  data.plot_tag,
                                                  data.title,
                                                  array_to_grid(convert_to_numpy(data.data),
                                                                data.images_per_row,
                                                                fill_value=np.NaN),
                                                  data.xmin,
                                                  data.xmax)
                    if data.tag == "text":
                        self.text_plotter.plot(environment_name,
                                               data.plot_tag,
                                               data.title,
                                               data.data)
