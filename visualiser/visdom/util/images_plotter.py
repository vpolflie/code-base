"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import math
import numpy as np

# Internal imports


class ImagesPlotter:

    """Plots to Visdom"""

    def __init__(self, viz):
        """
        Initialise an images plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}

    def plot(self, environment_name, plot_key, title, images, number_of_rows=1):
        """
        Plots a multiple images

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            images: image in numpy array form
            number_of_rows: number of rows in the image grid

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        # Insert the plot in the visdom environment or overwrite it
        padding = 10
        number_of_images_per_row = math.ceil(images.shape[0] / number_of_rows)
        height = number_of_images_per_row * (images.shape[2] + padding)
        width = number_of_rows * (images.shape[3] + padding)

        if plot_key not in self.plots[environment_name]:
            self.plots[environment_name][plot_key] = self.viz.images(images,
                                                                     env=environment_name,
                                                                     nrow=number_of_rows,
                                                                     padding=padding,
                                                                     opts={
                                                                       "caption": title,
                                                                       "width": width,
                                                                       "height": height,
                                                                     })
        else:
            self.viz.images(images,
                            win=self.plots[environment_name][plot_key],
                            env=environment_name,
                            nrow=number_of_rows,
                            padding=padding,
                            opts={
                                "caption": title,
                                "width": width,
                                "height": height,
                            }
                            )
