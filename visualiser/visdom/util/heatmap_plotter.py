"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np

# Internal imports


class HeatmapPlotter:

    """Plots to Visdom"""

    def __init__(self, viz):
        """
        Initialise an heatmap plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}

    def plot(self, environment_name, plot_key, title, heatmap, xmin=0, xmax=1):
        """
        Plot a single image

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            heatmap: image in numpy array form
            xmin: min displayed value
            xmax: max displayed value

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        # Insert the plot in the visdom environment or overwrite it
        # Add some extra width for the value bar
        heatmap = np.flip(heatmap, axis=1)
        width = heatmap.shape[2] + 100
        height = heatmap.shape[1]
        heatmap = heatmap[0]
        if plot_key not in self.plots[environment_name]:
            self.plots[environment_name][plot_key] = self.viz.heatmap(
                heatmap,
                env=environment_name,
                opts={
                  "width": width,
                  "height": height,
                  "xmax": xmax,
                  "xmin": xmin,
                  "caption": title,
                  "title": title
                })
        else:
            self.viz.heatmap(heatmap,
                             win=self.plots[environment_name][plot_key],
                             env=environment_name,
                             opts={
                                "width": width,
                                "height": height,
                                "xmax": xmax,
                                "xmin": xmin,
                                "caption": title,
                                "title": title
                             })
