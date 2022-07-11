"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports


class HistogramPlotter:

    """Plots to Visdom"""
    def __init__(self, viz):
        """
        Initialise an histogram plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}

    def plot(self, environment_name, plot_key, title, x, number_of_bins=30):
        """
        Plot Histograms given an array

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            x: data
            number_of_bins:

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        # Insert the plot in the visdom environment or overwrite it
        if plot_key not in self.plots[environment_name]:
            self.plots[environment_name][plot_key] = self.viz.histogram(
                X=x,
                env=environment_name,
                opts=dict(
                    title=title,
                    numbbins=number_of_bins
                ))
        else:
            self.viz.histogram(win=self.plots[environment_name][plot_key],
                               X=x,
                               env=environment_name,
                               opts=dict(
                                   title=title,
                                   numbbins=number_of_bins
                               ))
