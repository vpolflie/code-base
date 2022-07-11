"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np

# Internal imports


def normalize(X, mean, std):
    """
    Normalize an array given a channel first input

    Args:
        X: numpy array with a channel first format
        mean: numpy array with 1 or the same amount of elements as the amount of channels in X
        std: numpy array with 1 or the same amount of elements as the amount of channels in X
    """
    return (np.array(X) - np.array(mean)) / np.array(std)


def denormalize(X, mean, std, clip_value_min=None, clip_value_max=None, **kwargs):
    """
    Normalize an array given a channel first input

    Args:
        X: numpy array with a channel first format
        mean: numpy array with 1 or the same amount of elements as the amount of channels in X
        std: numpy array with 1 or the same amount of elements as the amount of channels in X

    Parameters:
        clip_value_min: The minimum allowed value in the array
        clip_value_max: The maximum allowed value in the array
    """
    # Get total shape
    shape = [1] * len(X.shape)
    # Get number of channels
    shape[1] = X.shape[1]

    # Denormalize
    denormalized_array = np.array(X) * np.array(std).reshape(shape) + np.array(mean).reshape(shape)
    if clip_value_min is not None or clip_value_max is not None:
        denormalized_array = np.clip(denormalized_array, clip_value_min, clip_value_max)
    return denormalized_array


def threshold(X, threshold, min_value=0., **kwargs):
    """
    Threshold an array. Everything below the threshold will be set to 0.

    Args:
         X: input numpy array
         threshold: float indicating the threshold

    Parameters:
        min_value: float, value to set all the thresholded values to
    """
    X[X < threshold] = min_value
    return X
