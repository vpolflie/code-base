"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
import config.config as cfg


class VisualisationData:
    
    """
    Object to save data visualisations, note that all data is expected in channel first format
    
    Args:
        tag: what kind of a visualisation [line, histogram, image, images, segmentation, segmentations, heatmap]
        plot_tag: tag to save the plot with
        title: title of the plot

    Parameters:
        text: optional option in some plots, should be a string
        data: list, numpy array, tensorflow or pytorch tensor
        moving_average_lengths: moving average of a line graph [line]
        number_of_bins: number of bins ina  histogram [histogram]
        store_history: boolean - store the history of all plotted images [image]
        images_per_row: Number of images per row to be displayed [images]
        xmin: Minimum value in the heatmap [heatmap]
        xmax: Maximum value in the heatmap [heatmap]
    """

    def __init__(self, tag, plot_tag, title, data,
                 text="",
                 moving_average_lengths=None,
                 number_of_bins=None,
                 store_history=None,
                 images_per_row=None,
                 xmin=None,
                 xmax=None,):
        self.tag = tag
        self.plot_tag = plot_tag
        self.title = title
        self.data = data if data is not None else None
        self.text = text

        self.moving_average_length = \
            moving_average_lengths if moving_average_lengths else cfg.VISUALISER.MOVING_AVERAGE_LENGTHS
        self.number_of_bins = number_of_bins if number_of_bins else cfg.VISUALISER.NUMBER_OF_BINS
        self.store_history = store_history if store_history else cfg.VISUALISER.STORE_HISTORY
        self.images_per_row = images_per_row if images_per_row else cfg.VISUALISER.IMAGES_PER_ROW
        self.xmin = xmin if xmin is not None else cfg.VISUALISER.XMIN
        self.xmax = xmax if xmax is not None else cfg.VISUALISER.XMIN
