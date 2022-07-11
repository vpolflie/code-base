"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Internal imports


import logging

log = logging.getLogger(__name__)


class SimpleInputScaler(BaseEstimator, TransformerMixin):
    """
    Simple Transformer wrapper for scaling an array

    Parameters:
    sf: float
        scale factor, typically 1/255. to map 8-bit RGBs -> [0, 1]
    """

    def __init__(self, sf=1 / 255.):
        self.sf = sf

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X * self.sf


class Float32er(BaseEstimator, TransformerMixin):
    """
    Simple Transformer wrapper for casting an array to 32-bit float
    """

    def fit(self, X):
        return self

    def transform(self, X):
        return X.astype(np.float32)


class Int32er(BaseEstimator, TransformerMixin):
    """
    Simple Transformer wrapper for casting an array to 32-bit float
    """

    def fit(self, X):
        return self

    def transform(self, X):
        return X.astype(np.int32)


class UInt8er(BaseEstimator, TransformerMixin):
    """
    Simple Transformer wrapper for casting an array to Uint8
    """

    def fit(self, X):
        return self

    def transform(self, X):
        return X.astype(np.uint8)
