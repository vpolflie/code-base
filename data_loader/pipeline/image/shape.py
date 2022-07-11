"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.transform import resize, rescale

# Internal imports
from tools.data.shape import pad
import logging

log = logging.getLogger(__name__)


class Padder(TransformerMixin, BaseEstimator):
    """
    Transformer to pad an image array up to a certain shape with constant values

    Pad an array at the end of each of the first two dimensions with constant values
    such that each of these is at least min_rows, min_cols respectively

    Args:
        width: int
            the minimum number of acceptable rows in the output
        height: int
            the minimum number of acceptable cols in the output


    Parameters:
        constant_values: int, optional the values to pad with
        mode: string - refers to the numpy padding options, mainly constant or reflect
        symmetric: bool, whether to spread the padding between either side of just on one singular side

    Notes:
    If used on an array from a raster, the choice of using the end of each dim
    will not shift the origin (top left), so the geotransform is unchanged
    """
    def __init__(self, width, height, constant_values=1, mode="constant", symmetric=False):
        self.width = width
        self.height = height
        self.constant_values = constant_values
        self.symmetric = symmetric
        self.mode = mode

    def fit(self, X):
        return self

    def transform(self, X):
        padding_width = max(self.width - X.shape[0], 0)
        padding_height = max(self.height - X.shape[1], 0)
        if self.symmetric:
            pad_parameters = ((padding_width // 2, padding_width // 2 + padding_width % 2),
                              (padding_height // 2, padding_height // 2 + padding_height % 2),
                              (0, 0))
        else:
            pad_parameters = ((0, padding_width), (0, padding_height), (0, 0))

        if self.mode == "constant":
            return pad(X, pad_parameters, mode=self.mode, constant_values=self.constant_values)
        else:
            return pad(X, pad_parameters, mode=self.mode)


class Resizer(TransformerMixin, BaseEstimator):
    """
    Transformer for applying skimage.rescale to an image-like array

    Attributes:
            target_resolution: int
                Resampling factor: 0.5 => half spatial resolution
            preserve_int: bool, optional
                Ensure that 8-bit integer inputs produce (rounded) 8-bit integer outputs
    """

    def __init__(self, target_resolution, preserve_int=True, is_mask=False, binary_mask_fraction=0.5):
        self.target_resolution = target_resolution
        if isinstance(self.target_resolution, int):
            self.target_resolution = (self.target_resolution, self.target_resolution)
        self.preserve_int = preserve_int
        self.is_mask = is_mask
        self.binary_mask_fraction = binary_mask_fraction

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Parameters:
        X: np.ndarray
            Input array (upon which fit was called)

        Returns:
            np.ndarray
                Resized array
        """
        resized = resize(X, self.target_resolution, anti_aliasing=True)
        if self.is_mask:
            resized[resized > self.binary_mask_fraction] = 1.
            resized[resized <= self.binary_mask_fraction] = 0.
        if X.dtype == np.uint8 and self.preserve_int:
            resized = resized.astype(np.uint8)
        return resized


class Rescaler(TransformerMixin, BaseEstimator):
    """
    Transformer for applying skimage.rescale to an image-like array

    Attributes:
            target_resolution: int
                Resampling factor: 0.5 => half spatial resolution
            preserve_int: bool, optional
                Ensure that 8-bit integer inputs produce (rounded) 8-bit integer outputs
    """

    def __init__(self, target_resolution, preserve_int=True, is_mask=False, binary_mask_fraction=0.5):
        self.target_resolution = target_resolution
        if isinstance(self.target_resolution, int):
            self.target_resolution = (self.target_resolution, self.target_resolution)
        self.preserve_int = preserve_int
        self.is_mask = is_mask
        self.binary_mask_fraction = binary_mask_fraction

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Parameters:
        X: np.ndarray
            Input array (upon which fit was called)

        Returns:
            np.ndarray
                Resized array
        """
        scale = min((self.target_resolution[0] / X.shape[0], self.target_resolution[1] / X.shape[1]))
        rescaled = rescale(X, scale, anti_aliasing=True)
        if self.is_mask:
            rescaled[rescaled > self.binary_mask_fraction] = 1.
            rescaled[rescaled <= self.binary_mask_fraction] = 0.
        if X.dtype == np.uint8 and self.preserve_int:
            rescaled = rescaled.astype(np.uint8)
        return rescaled


class Cropper(BaseEstimator, TransformerMixin):
    """
    Transformer to crop an image-like array (along the first two axes)

    The output is the subset of the input which fits into an integer multiple
    of window_dims.

    Parameters:
    window_dims: array_like, optional
        h, w of eventual window/tile size
    center: always crop the center
    """

    def __init__(self, window_dims=(224, 224), center=False):
        """ defaults to adding +/- 2 pixels in each dimension """
        self.window_dims = window_dims
        self.center = center

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Parameters:
        X: np.ndarray
            Input (H, W, C) array (upon which fit was called)

        Returns:
        np.ndarray
            Cropped (H', W', C) array; H' <= H, W' <= W
        """
        if not self.center:
            x_start, y_start = 0, 0
            if X.shape[0] > self.window_dims[0]:
                x_start = random.randint(0, X.shape[0] - self.window_dims[0] - 1)
            if X.shape[1] > self.window_dims[1]:
                y_start = random.randint(0, X.shape[1] - self.window_dims[1] - 1)
        else:
            x_start = (X.shape[0] - self.window_dims[0]) // 2
            y_start = (X.shape[1] - self.window_dims[1]) // 2
        X_ = X[x_start:x_start + self.window_dims[0], y_start:y_start + self.window_dims[1]]
        return X_


class DimensionAdder(BaseEstimator, TransformerMixin):
    """
    Simple Transformer to add a channel dimension to an array if it's 2D

    For example, array with shape (H, W) -> (H, W, 1)
    """

    def fit(self, X):
        return self

    def transform(self, X):
        return X[..., np.newaxis]


class ChannelSelector(TransformerMixin, BaseEstimator):
    """
    Transformer to select channels / project out slices along last axis of array

    Parameters:
    channels: array_like, optional
        Indices of channels to select
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X):
        return self

    def transform(self, X):
        return X[..., self.channels]


class NoneChannelAdder(TransformerMixin, BaseEstimator):
    """
    Transformer to add masked channel which indicates no presence of information
    """

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Given an array in the following format WxHxC, where each channel can have the value 0 and 1 and where
        the sum over C doesnt necessarily add up to 1. Add a new channel at the start which represents an absent class.

        Args:
            X: np array
        """
        X = np.concatenate((1 - np.clip(np.sum(X, axis=-1, keepdims=True), 0, 1), X), axis=-1)
        return X


class ChannelFirstToLast(TransformerMixin, BaseEstimator):
    """
    Transformer which switches a tensor from channel first to last
    """

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Assuming CxHxW transform to WxHxC

        Args:
            X: input np array
        """
        X = np.moveaxis(X, -1, 0)
        return X


class ChannelLastToFirst(TransformerMixin, BaseEstimator):
    """
    Transformer which switches a tensor from channel last to first
    """

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Assuming WxHxC transform to CxHxW

        Args:
            X: input np array
        """
        X = np.moveaxis(X, -1, 0)
        return X

