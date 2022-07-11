"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import random

# Internal imports


"""
    Fancy pca module for smarter data augmentation
"""
# TODO provide script to generate FancyPCA components


def channel_cov_eigh(X):
    """
    Obtain the eigendecomposition of the channel data of an image array

    Necessary step for the so-called "Fancy PCA" image augmentation technique

    Parameters:
    X: array_like
        A NumPy or Dask array with >= 2 dimensions, the last corresponding to
        the channels whose eigendecomposition will be calculated

    Returns:
    tuple
        (eigenvalues, eigenvectors) sorted in decreasing order of eval magnitude
    """
    # work with dask
    lib= np
    # flatten to shape (N_samples*H*W , 3 (RGB)), cast to float
    X_flat = X.reshape(-1, X.shape[-1])
    if X.dtype == np.uint8:
        X_flat /= 255.
    # centre
    X_mean = X_flat.mean(axis=0)
    X_ = X_flat - X_mean
    # calculate covariance matrix
    R = lib.cov(X_, rowvar=False)
    # compute 3x3 matrix for dask
    if hasattr(R, 'compute'):
        R = R.compute()
    # eigendecomposition
    eig_vals, eig_vecs = np.linalg.eigh(R)
    # sort eigenvectors by value and vectors correspondingly
    sort_perm = eig_vals[::-1].argsort()
    eig_vecs = eig_vecs[:, sort_perm]
    eig_vals = eig_vals[sort_perm]
    return eig_vals, eig_vecs


def sample_channel_shift(eig_vals, eig_vecs, alpha_std=0.1):
    """
    Generate a random channel shift vector to be added to each pixel of an image

    Produces a linear combination of random variable alpha (sampled from normal
    distn centred at 0 with std alpha_std) * cov eigenvalue * cov eigenvector

    Notes:
    Used in fancy PCA to apply data-derived channel shifts in RGB space

    Parameters:
    eig_vals: array_like
        eigenvalues of the channel covariance matrix sorted in decreasing order
        of magnitude
    eig_vecs: array_like
        the channel-space eigenvectors to which of eig_vals applies
    alpha_std: float
        the stdev of the gaussian used to sample the eigenvalue magnitudes used
        in generating the shift


    Returns:
    pert_vector: array_like
        an (n_channels,) shaped array containing the channel-perturbation
    """
    n_channels = eig_vecs.shape[0]
    # get 3x1 matrix of eigenvalues multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of alpha_std
    pert_magnitudes = np.zeros((n_channels, 1))
    # draw only once per augmentation
    alpha = np.random.normal(0, alpha_std, (n_channels,))
    # broadcast
    pert_magnitudes[:, 0] = alpha * eig_vals[:]
    # this is the vector that we're going to add to each pixel
    pert_vector = np.dot(eig_vecs.T, pert_magnitudes).T
    return pert_vector


class PCAChannelShiftSampler:
    """
    Callable which generates shift vectors with "Fancy PCA" sampling for images

    This derives the PCA decomposition of the channels from the input array, and
    acts as a callable which generates random RGB shift vectors according to
    this decomposition

    Parameters:
    X: array_like
        An image-like array with channels along the last axis. Used to calculate
        shift vectors based on the eigendecomposition of the covariance matrix.
    alpha_std: float, optional
        The stdev of the gaussian used to sample the channel shifts according to
        the eigendecompostion. wider => bigger perturbations.

    Attributes:
    eig_vals: array_like
        the channel covariance matrix's eigenvalues in desc order of magnitude
    eig_vecs: array_like
        the channel covariance matrix's eigenvectors in the same order as evals

    Returns
    -------
    array_like
        an (n_channels,) shaped random shift vector
    """

    def __init__(self, X, alpha_std=0.1, max_samples=5000):
        self.X = X
        self.max_samples = max_samples
        self.alpha_std = alpha_std
        self.calculate_eigenvectors()

    def calculate_eigenvectors(self):
        if self.max_samples:
            self.eig_vals, self.eig_vecs = channel_cov_eigh(self.X[:self.max_samples])
        else:
            self.eig_vals, self.eig_vecs = channel_cov_eigh(self.X)

    def __call__(self):
        return sample_channel_shift(eig_vals=self.eig_vals,
                                    eig_vecs=self.eig_vecs,
                                    alpha_std=self.alpha_std)

    def __repr__(self):
        return f'PCAChannelShiftSampler({self.X}, alpha_std={self.alpha_std})'


def apply_channel_shift(img, sampler: PCAChannelShiftSampler):
    """
    Generates a random channel shift vector and applies it to an image array

    Uses 'fancy PCA' sampler and perturbs input image array in channel space

    Arguments:
    img: array_like
        input array with channels along the last dimension
    sampler: PCAChannelShiftSampler
        a sampler instance

    Returns:
    array_like
        input image + sampled perturbation, applied globally to all pixels
    """
    shift = sampler()
    if img.dtype == np.uint8:
        img_out = img + 255. * shift
        img_out = np.clip(img_out, 0.0, 255.0)
        return np.rint(img_out).astype(np.uint8)
    else:
        return np.clip(img + shift, 0.0, 255.0)


class FancyPCA(ImageOnlyTransform):
    """
    Augment RGB image using FancyPCA

    Parameters:
    alpha_std: float
        Sampling gaussian perturbation stdev. See sample_channel_shift.
    always_apply: bool, optional
        Whether to always apply.
        See albumentations.core.transforms_interface.ImageOnlyTransform

    Attributes:
    sampler: PCAChannelShiftSampler
        Sampler instance used to derive channel shift vectors internally

    Notes:
    from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    References:
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
    https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, X, alpha_std=0.3, always_apply=False, p=0.5, max_samples=1000):
        super(FancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha_std = alpha_std
        self.sampler = PCAChannelShiftSampler(X, self.alpha_std, max_samples=max_samples)

    @property
    def alpha_std(self):
        return self._alpha_std

    @alpha_std.setter
    def alpha_std(self, value):
        self._alpha_std = value
        if hasattr(self, 'sampler'):
            self.sampler.alpha_std = value

    def apply(self, img, alpha=0.1, **params):
        return apply_channel_shift(img, self.sampler)

    def get_params(self):
        return {"alpha_std": random.gauss(0, self.alpha_std)}

    def get_transform_init_args_names(self):
        return ("alpha_std",)
