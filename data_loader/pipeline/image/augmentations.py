"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import albumentations
import numpy as np
import random
from skimage.transform import rotate, rescale
from sklearn.base import BaseEstimator, TransformerMixin

# Internal imports


def spatial_augmentations(
        probability=.8, probability_rotations_90=0., probability_horizontal_flip=.5,
        probability_iaa_affine=.01, probability_shift_scale_rotate=.2, probability_distort=.02,
):
    """
    Returns a composition of albumentations spatial transformations

    Parameters:
    probability : float, optional
        Global augmentation probability in [0,1.] (of doing any augs at all)
    probability_rotations_90 :
        Probability of random rotation by multiple of 90 degrees
    probability_horizontal_flip :
        Horizontal flip probability
    probability_iaa_affine :
        Probability of affine transformation
    probability_shift_scale_rotate :
        Probability of shift-scale-rotate
    probability_distort :
        Probability of distortion

    Returns:
    albumentations.core.composition.Compose
        A composition of various albumentations transformations for
        image augmentation
    """
    aug_list = []
    # Apply 90 degree rotations
    if probability_rotations_90:
        aug_list.append(albumentations.RandomRotate90(p=probability_rotations_90))
    # Apply an horizontal flip
    if probability_horizontal_flip:
        aug_list.append(albumentations.HorizontalFlip(p=probability_horizontal_flip))
    # Apply affine transformation
    if probability_iaa_affine:
        aug_list.append(
            albumentations.IAAAffine(scale=1.0,
                                     translate_percent=(-0.1, 0.1),
                                     translate_px=None,
                                     rotate=0.0,
                                     shear=(-10, 10),
                                     order=1,
                                     cval=0,
                                     mode='reflect',
                                     always_apply=False,
                                     p=probability_iaa_affine)
        )
    # Apply a shift, scale and rotate on the image
    if probability_shift_scale_rotate:
        aug_list.append(
            albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                            scale_limit=0.1,
                                            rotate_limit=45,
                                            p=probability_shift_scale_rotate)
        )
    # Apply affine transformation
    if probability_distort:
        aug_list.append(
            albumentations.OneOf([
                albumentations.IAAPiecewiseAffine(p=1.0),
            ], p=probability_distort)
        )
    return albumentations.Compose(aug_list, p=probability)


def color_augmentations(
        probability=.8, fancy_pca=None, probability_rgb_shift=.4, probability_noise=.2, probability_blur=.02,
        probability_sharpen=0.2, probability_brightness_contrast=0.4, probability_hue_sat=0.05, probability_gamma=.1
):
    """
    Returns a composition of albumentations color transformations

    Parameters:
    probability : float, optional
        Global augmentation probability in [0,1.] (of doing any augs at all)
    fancy_pca : :obj:`FancyPCA`, optional
        An instance of FancyPCA, evaluated on the training dataset
        with precalculated shift eigenvectors
    probability_rgb_shift :
        Probability of random RGB shift
    probability_noise :
        Probability of gaussian noise
    probability_blur :
        Probability of gaussian blur
    probability_sharpen :
        Probability of sharpening
    probability_brightness_contrast :
        Probability of contrast shift
    probability_hue_sat :
        Probability of hue-saturation shift
    probability_gamma :
        Probability of gamma shift

    Returns:
    albumentations.core.composition.Compose
        A composition of various albumentations transformations for
        image augmentation
    """
    aug_list = []
    # Apply Fancy PCA
    if fancy_pca is not None:
        aug_list.append(fancy_pca)
    # Apply RGB shift
    if probability_rgb_shift:
        aug_list.append(albumentations.RGBShift(p=probability_rgb_shift,
                                                r_shift_limit=(-15, 15),
                                                g_shift_limit=(-15, 15),
                                                b_shift_limit=(-15, 15)))
    # Apply noise on the image
    if probability_noise:
        aug_list.append(
            albumentations.OneOf([
                albumentations.IAAAdditiveGaussianNoise(),
                albumentations.GaussNoise(var_limit=(5., 20.)),
            ], p=probability_noise)
        )
    # Apply blur the image
    if probability_blur:
        aug_list.append(
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=(3, 4), p=1.),
                albumentations.Blur(blur_limit=(3, 4), p=1.),
            ], p=probability_blur)
        )
    # Sharpen image
    if probability_sharpen:
        aug_list.append(albumentations.IAASharpen(p=probability_sharpen))
    # Change brightness and contrast of image
    if probability_brightness_contrast:
        aug_list.append(albumentations.RandomBrightnessContrast(p=probability_brightness_contrast))
    # Change Hue and Saturation of image
    if probability_hue_sat:
        aug_list.append(
            albumentations.HueSaturationValue(10, 10, 10, p=probability_hue_sat)
        )
    # Change Gamma of image
    if probability_gamma:
        aug_list.append(
            albumentations.RandomGamma(gamma_limit=(90, 110), p=probability_gamma)
        )
    return albumentations.Compose(aug_list, p=probability)


class CopyPasteAugmenter(TransformerMixin, BaseEstimator):
    """
    Cut out specific parts out of an image and add these to a different image
    """

    def __init__(self, min_rescale_factor=0.8, max_rescale_factor=1.2, min_rotate_factor=0, max_rotate_factor=360):
        self.min_rescale_factor = min_rescale_factor
        self.max_rescale_factor = max_rescale_factor
        self.min_rotate_factor = min_rotate_factor
        self.max_rotate_factor = max_rotate_factor

    def fit(self, X):
        return self

    def transform(self, copy_X, copy_Y, paste_X, paste_Y):
        # Rotate and Rescale
        rescale_factor = random.random() * (self.max_rescale_factor - self.min_rescale_factor) + self.min_rescale_factor
        rotate_factor = random.random() * (self.max_rotate_factor - self.min_rotate_factor) + self.min_rotate_factor
        copy_Y = rescale(copy_Y, (rescale_factor, rescale_factor, 1))
        copy_X = rescale(copy_X, (rescale_factor, rescale_factor, 1))
        copy_Y = rotate(copy_Y, rotate_factor)
        copy_X = rotate(copy_X, rotate_factor)

        # Copy
        boundaries = np.argwhere(copy_Y)
        (boundary_x_start, boundary_y_start, _), (boundary_x_end, boundary_y_end, _) = \
            boundaries.min(0), boundaries.max(0) + 1
        copied_X = copy_X[boundary_x_start:boundary_x_end, boundary_y_start:boundary_y_end]
        copied_Y = copy_Y[boundary_x_start:boundary_x_end, boundary_y_start:boundary_y_end]

        # Get paste indices
        copied_x_index = 0
        paste_x_index = 0
        if copied_X.shape[0] > paste_X.shape[0]:
            copied_x_index = random.randrange(0, copied_X.shape[0] - paste_X.shape[0])
            x_size = paste_X.shape[0]
        elif copied_X.shape[0] < paste_X.shape[0]:
            paste_x_index = random.randrange(0, paste_X.shape[0] - copied_X.shape[0])
            x_size = copied_X.shape[0]
        else:
            x_size = paste_X.shape[0]

        copied_y_index = 0
        paste_y_index = 0
        if copied_X.shape[1] > paste_X.shape[1]:
            copied_y_index = random.randrange(0, copied_X.shape[1] - paste_X.shape[1])
            y_size = paste_X.shape[1]
        elif copied_X.shape[1] < paste_X.shape[1]:
            paste_y_index = random.randrange(0, paste_X.shape[1] - copied_X.shape[1])
            y_size = copied_X.shape[1]
        else:
            y_size = paste_X.shape[1]

        # Get alpha_masks
        alpha_mask = np.zeros(paste_Y.shape, dtype=np.float32)
        alpha_mask[paste_x_index:paste_x_index + x_size, paste_y_index:paste_y_index + y_size] = \
            copied_Y[copied_x_index:copied_x_index + x_size, copied_y_index:copied_y_index + y_size]
        rgb_copy = np.zeros(paste_X.shape, dtype=np.float32)
        rgb_copy[paste_x_index:paste_x_index + x_size, paste_y_index:paste_y_index + y_size] = \
            copied_X[copied_x_index:copied_x_index + x_size, copied_y_index:copied_y_index + y_size]

        # Paste
        X = rgb_copy * alpha_mask * 255 + paste_X * (1 - alpha_mask)
        Y = np.logical_or(alpha_mask, paste_Y).astype(np.float32)

        return X, Y

