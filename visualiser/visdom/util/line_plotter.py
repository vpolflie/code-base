"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np

# Internal imports


class LinePlotter:
    """Plots to Visdom"""

    def __init__(self, viz):
        """
        Initialise an image plotter

        Args:
            viz: visdom object
]        """
        self.viz = viz
        self.plots = {}
        self.history = {}

    def plot(self, environment_name, plot_key, title, x, y, variable_name, moving_average_lengths=(50,)):
        """
        Plot a line graph

        Args:
            environment_name: visdom environment name
            plot_key: the plot key of the dictionary
            title: title of the plot
            x: current iteration
            y: value
            variable_name: variable you want to plot
            moving_average_lengths: a list or tuple of moving averages you want to plot

        Returns:

        """
        # Create a new visdom environment if it doesnt exist yet
        if environment_name not in self.plots:
            self.plots[environment_name] = {}

        # Insert the plot in the visdom environment or overwrite it
        # If the moving averages plot is initialised store the data and plot a moving average
        if plot_key not in self.plots[environment_name]:
            if environment_name not in self.history:
                self.history[environment_name] = {}

            self.history[environment_name][plot_key] = [y]
            self.plots[environment_name][plot_key] = self.viz.line(
                X=[x],
                Y=[y],
                env=environment_name,
                opts=dict(
                    legend=[variable_name],
                    title=title,
                    xlabel='Iterations',
                    ylabel=variable_name
                ))

            for moving_average in moving_average_lengths:
                self.viz.line(win=self.plots[environment_name][plot_key],
                              X=[x],
                              Y=[y],
                              env=environment_name,
                              name="moving average " + variable_name + " %d" % moving_average,
                              update='append')
        else:
            if environment_name not in self.history:
                self.history[environment_name] = {}

            if plot_key not in self.history[environment_name]:
                self.history[environment_name][plot_key] = [y]
            else:
                self.history[environment_name][plot_key].append(y)

            self.viz.line(win=self.plots[environment_name][plot_key],
                          X=[x],
                          Y=[y],
                          env=environment_name,
                          name=variable_name,
                          update='append')

            for moving_average in moving_average_lengths:
                y_average = np.mean(self.history[environment_name][plot_key][-moving_average:])

                self.viz.line(win=self.plots[environment_name][plot_key],
                              X=[x],
                              Y=[y_average],
                              env=environment_name,
                              name="moving average " + variable_name + " %d" % moving_average,
                              update='append')
