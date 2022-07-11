"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports


class TextPlotter:

    """Plots to Visdom"""

    def __init__(self, viz):
        """
        Initialise an image plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}

    def plot(self, environment_name, plot_key, title, text):
        """
        Plot a some text

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            text: string to be displayed

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        text_field = title + "\n" + ("-" * len(title) + "\n") + text

        # Insert the plot in the visdom environment or overwrite it
        if plot_key not in self.plots[environment_name]:
            self.plots[environment_name][plot_key] = self.viz.text(
                text_field,
                env=environment_name)
        else:
            self.viz.image(text_field,
                           win=self.plots[environment_name][plot_key],
                           env=environment_name)
