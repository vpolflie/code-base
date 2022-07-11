"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import math
import numpy as np


SEGMENTATION_CHANNELS_IMAGE_CODING = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
    [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128], [128, 128, 128],
    [192, 0, 0], [0, 192, 0], [0, 0, 192], [192, 192, 0], [0, 192, 192], [192, 0, 192], [192, 192, 192],
    [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0], [0, 64, 64], [64, 0, 64], [64, 64, 64], [255, 255, 255],
])


def list_of_dict_2_dict_of_list(list_of_dict):
    """
        Transforms a list of dictionaries to a dictionary with lists

        Args:
            list_of_dict: a list of dictionaries

        Returns:
            A dictionary containing lists
    """
    dict_of_list = {}
    for dictionary in list_of_dict:
        for k, v in dictionary.items():
            if k in dict_of_list:
                dict_of_list[k].append(v)
            else:
                dict_of_list[k] = [v]

    return dict_of_list


def array_to_segmentation(arrays, **kwargs):
    """
    Convert arrays of shape BxCxHxW to Bx3xHxW

    Args:
        arrays: a numpy array

    Returns: a numpy array with 3 channels
    """
    assert 3 <= len(arrays.shape) < 5, "This function is only designed for arrays of shape BxCxHxW or CxHxW"

    # Check if there isn't a batch channel
    if len(arrays.shape) < 4:
        arrays = arrays[None]

    # Fix every array
    _arrays = []
    for x in arrays:
        # If only one channel convert it to 3
        if x.shape[0] == 1:
            x = np.concatenate((1 - x, x), axis=0)

        x = np.argmax(x, axis=0)[None]
        _arrays.append(x)
    _arrays = np.array(_arrays)

    return _arrays


def array_to_segmentation_image(arrays, **kwargs):
    """
    Convert arrays of shape BxCxHxW to color coded Bx3xHxW representation

    Args:
        arrays: a numpy array

    Returns: a numpy array with 3 channels
    """
    assert 3 <= len(arrays.shape) < 5, "This function is only designed for arrays of shape BxCxHxW or CxHxW"

    # Check if there isn't a batch channel
    if len(arrays.shape) < 4:
        arrays = arrays[None]

    # Check if there is a specific color coding specified
    if "color_coding" in kwargs and kwargs["color_coding"] is not None:
        if len(kwargs["color_coding"]) == arrays.shape[0]:
            color_coding = kwargs["color_coding"][0]
        else:
            color_coding = kwargs["color_coding"]
    else:
        color_coding = SEGMENTATION_CHANNELS_IMAGE_CODING

    # Check if color coding is large enough for the number of channels
    assert len(color_coding) > arrays.shape[1]

    # Fix every array
    images = []
    for x in arrays:
        # If only one channel convert it to 3
        if x.shape[0] == 1:
            x = np.concatenate((1-x, x), axis=0)

        argmax_x = np.argmax(x, axis=0)
        _x = np.zeros((x.shape[1], x.shape[2], 3))
        for i in range(x.shape[0]):
            _x[argmax_x == i] = color_coding[i] / 255
        x = _x
        x = np.moveaxis(x, 2, 0)

        images.append(x)

    # Convert the images to a numpy array
    images = np.array(images)

    return images


def array_to_image(arrays,  **kwargs):
    """
    Convert arrays of shape BxCxHxW to Bx3xHxW

    Args:
        arrays: a numpy array
        grid: boolean to convert the image to a grid

    Returns: a numpy array with 3 channels
    """
    assert 3 <= len(arrays.shape) < 5, "This function is only designed for arrays of shape BxCxHxW or CxHxW"

    # Check if there isn't a batch channel
    if len(arrays.shape) < 4:
        arrays = arrays[None]

    # This functions is only for gray scale / rgb images

    assert arrays.shape[1] == 1 or arrays.shape[1] == 3

    # Fix every array
    images = []
    for index, array in enumerate(arrays):
        # If only one channel convert it to 3
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        images.append(array)

    return np.array(images)


def array_to_grid(arrays, images_per_row=None, padding=10, fill_value=1.0, mirror=False, **kwargs):
    """
    Convert a batch of images to a single grid

    Args:
        arrays: a batch of numpy arrays
        images_per_row: the amount of images per row
        padding: padding in between the images
        fill_value: fill value for padding
        mirror: mirror the images along the y axis

    Returns: a single numpy array
    """
    # if images per row is not specified put them all on one row
    if arrays.shape[0] == 1:
        return arrays[0]

    if images_per_row is None:
        images_per_row = arrays.shape[0]

    # Calculate the number of rows
    number_of_rows = math.ceil(arrays.shape[0] / images_per_row)

    # Calculate width and height
    height = number_of_rows * arrays.shape[2] + padding * (number_of_rows - 1)
    width = images_per_row * arrays.shape[3] + padding * (images_per_row - 1)

    # Create new numpy array
    grid = np.full((height, width, arrays.shape[1]), fill_value=[fill_value] * arrays.shape[1])
    grid = np.moveaxis(grid, 2, 0)
    for index, image in enumerate(arrays):
        y_start = (image.shape[1] + padding) * math.floor(index / images_per_row)
        x_start = (image.shape[2] + padding) * (index % images_per_row)
        if mirror:
            image = np.flip(image, axis=1)
        grid[:, y_start:y_start+image.shape[1], x_start:x_start+image.shape[2]] = image

    return grid


def pad(array, sizes, mode="reflect", **kwargs):
    """
    Function that pads an array with a certain amount of rows/columns on a certain size
    Args:
        array: numpy array
        sizes: the amount that needs to be padded on each side (top, left, bottom, right)
        mode: type of padding

    Returns:
        np array
    """
    new_array = np.pad(array, sizes, mode, **kwargs)
    return new_array
