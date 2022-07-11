"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports

# Internal imports


class ImagePlotter:

    """Plots to Visdom"""

    def __init__(self, viz):
        """
        Initialise an image plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}

    def plot(self, environment_name, plot_key, title, image, store_history=False):
        """
        Plot a single image

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            image: image in numpy array form
            store_history: store the history of all the images in the plot with a slider

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        # Insert the plot in the visdom environment or overwrite it
        width = image.shape[2]
        height = image.shape[3]
        if plot_key not in self.plots[environment_name]:
            self.plots[environment_name][plot_key] = self.viz.image(
                image,
                env=environment_name,
                store_history=store_history,
                caption=title,
                opts={
                  "width": width,
                  "height": height,
                })
        else:
            self.viz.image(image,
                           win=self.plots[environment_name][plot_key],
                           env=environment_name,
                           store_history=store_history,
                           caption=title,
                           opts={
                               "width": width,
                               "height": height,
                           })
