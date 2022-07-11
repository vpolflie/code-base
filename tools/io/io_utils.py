"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread, imsave
import threading
import yaml

# Internal imports
from tools.data.shape import array_to_image, array_to_grid, array_to_segmentation_image
from tools.data.type import convert_to_numpy, convert_to_serializable
from tools.data.value import denormalize, threshold
from tools.path_utils import create_folders


def read_yaml(file_path):
    """
    Read a yaml file

    Args:
        file_path: path to the yaml file

    Returns:
        python dictionary containing the content of the yaml
    """

    # Create a data set object for each required data set
    # load the yml file as a dict
    with open(file_path, 'r') as f:
        try:
            yaml_dictionary = yaml.safe_load(f)
            return yaml_dictionary
        except Exception as e:
            raise e


def read_json(file_path):
    """
    Read a json file

    Args:
        file_path: path to the json file

    Returns:
        python dictionary containing the content of the json
    """

    # Create a data set object for each required data set
    # load the yml file as a dict
    with open(file_path, 'r') as f:
        return json.load(f)


def read_image(file_path):
    """
    Read an image file
    Args:
        file_path: path to the image file

    Returns:
        A numpy array representing the image
    """
    return imread(file_path)


def read_npy(file_path):
    """
    Read a numpy binary
    Args:
        file_path: path to the numpy binary

    Returns: a numpy array

    """
    return np.load(file_path, allow_pickle=True)


def write_image(file_path, image, **kwargs):
    """
    Write an image to the specified path
    Args:
        file_path: path to which the image should be written to
        image: the image

    Returns:

    """
    if "normalize" in kwargs.keys() and kwargs["normalize"]:
        image = denormalize(image, **kwargs)

    image = array_to_image(convert_to_numpy(image), **kwargs)
    image = array_to_grid(image, **kwargs)
    image = np.moveaxis((image * 255).astype(np.uint8), 0, 2)
    imsave(file_path, image, check_contrast=False)


def write_segmentation_image(file_path, image, **kwargs):
    """
    Write an image to the specified path
    Args:
        file_path: path to which the image should be written to
        image: the image

    Returns:

    """
    if "threshold" in kwargs.keys() and kwargs["threshold"] is not None:
        image = threshold(image, kwargs["threshold"])

    image = array_to_segmentation_image(convert_to_numpy(image), **kwargs)
    image = array_to_grid(image, **kwargs)
    image = np.moveaxis((image * 255).astype(np.uint8), 0, 2)
    imsave(file_path, image, check_contrast=False)


def write_heatmap(file_path, data, **kwargs):
    """
    Write an image to the specified path
    Args:
        file_path: path to which the image should be written to
        data: the heatmap

    Returns:

    """
    data = convert_to_numpy(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = array_to_grid(data, fill_value=np.NaN)
    data = np.moveaxis((data * 255).astype(np.uint8), 0, 2)
    imsave(file_path, data, check_contrast=False)


def write_json(file_path, data, **kwargs):
    """
    Ditch specific data
    Args:
        file_path: path to which the image should be written to
        data: data to be ditched
    Returns:
    """
    with open(file_path, 'w') as f:
        return json.dump(convert_to_serializable(data), f)


def write_none(file_path, data, **kwargs):
    """
    Ditch specific data
    Args:
        file_path: path to which the image should be written to
        data: data to be ditched
    Returns:
    """
    pass


def dump_visualisations(path, data, **kwargs):
    """
    Function to dump VisualisationData objects to a specific folder
    Args:
        path: path to folder
        data: a list of VisualisationData objects

    Returns:

    """
    # Create the save path
    save_path = os.path.join(path, "visualisations")
    create_folders(save_path)

    # Save each data object
    # Collect non image data and save image data
    json_dict = {}
    for data_object in data:
        if data_object.data is not None:
            if data_object.tag == "image":
                write_image(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "images":
                write_image(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "segmentation":
                write_segmentation_image(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "segmentations":
                write_segmentation_image(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "heatmap":
                write_heatmap(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "images":
                write_image(
                    os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"),
                    data_object.data,
                    **kwargs
                )
            if data_object.tag == "line":
                json_dict[data_object.plot_tag + "_" + data_object.plot_tag] = str(convert_to_numpy([data_object.data]))

            if data_object.tag == "histogram":
                plt.hist(
                    convert_to_numpy(data_object.data),
                    bins=data_object.number_of_bins if data_object.number_of_bins else 10,
                )
                plt.savefig(os.path.join(save_path, data_object.plot_tag + "_" + data_object.plot_tag + ".png"))

    # Save image data
    with open(os.path.join(save_path, "data.json"), "w") as f:
        json.dump(json_dict, f)


def dump_config(path, config):
    """
    Function to dump config data to a specific folder
    Args:
        path: path to folder
        config: a config namespace

    Returns:

    """
    dump_dict = {k.lower(): v for k, v in config.__dict__.items()}

    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(dump_dict, f)


def load_config(path, config):
    """
    Function to read a config yaml and update the current one
    Args:
        path: path to folder
        config: a config namespace

    Returns:

    """
    # Load yaml
    config_dict = read_yaml(path)
    for k, v in config_dict.items():
        if k != "load":
            # update config
            setattr(config, k.upper(), config_dict[k])

    return config


class OutputWriterThread(threading.Thread):

    def __init__(self, queue, write_methods, folders, extensions, thread_id=0):
        """
        A separate threading class to write data to disk

        Args:
            queue: a threading queue
            write_methods: list of function that will be used to write the data disk
            folders: list of folder names for the data
            extensions: list of extensions for the data
            thread_id: an identifier for the thread
        """
        super(OutputWriterThread, self).__init__()
        self.queue = queue
        self.thread_id = thread_id
        self.write_methods = write_methods
        self.folders = folders
        self.extensions = extensions
        assert not (self.folders == self.extensions and self.folders == self.write_methods), \
            "OutputWriterThread wrong configuration: writer_functions %d folders %d extensions %d" % (
                len(self.write_methods), len(self.folders), len(self.extensions)
            )

        # Create folders
        for folder in self.folders:
            create_folders(os.path.join("results", folder))

        # Threading parameter
        self.daemon = True
        self.done = False

    def run(self):
        """
        Threading runner, gets information and visualises it
        """
        pass
        while not self.done or self.queue.qsize() > 0:
            if self.queue.qsize() > 0:
                data, meta_data = self.queue.get()
                for index, data_entry in enumerate(data):
                    save_path = os.path.join("results",
                                             self.folders[index], meta_data["file_name"] + self.extensions[index])
                    self.write_methods[index](save_path, data_entry, **meta_data)
                self.queue.task_done()

    def signal_end(self):
        """
        Signal this thread that no more data will be incoming
        """
        self.done = True
