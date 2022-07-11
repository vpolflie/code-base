"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from sklearn.base import BaseEstimator, TransformerMixin

# Internal imports
from tools.data.value import normalize
import logging

log = logging.getLogger(__name__)


class Normalize(TransformerMixin, BaseEstimator):
    """
    Normalize an input according to a mean and standard deviation

    Args:
        mean: float or np array of floats, the mean of the overall data set. This array should have on single
              element or the same amount of elements as there are channels in the input
        std: float or np array of floats, the standard deviation of the overall data set. This array should have
            on single element or the same amount of elements as there are channels in the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def fit(self, X):
        return self

    def transform(self, X):
        return normalize(X, self.mean, self.std)
