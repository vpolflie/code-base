"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import argparse
import cv2
import glob
import numpy as np
import os

# Internal imports


def main(input_directories, extensions):
    """
    Function which calculates an approximation of the mean and variance given a series of input directories containing
    images,

    Args:
        input_directories: a list of paths to image directories
        extensions: a list of image extensions

    Returns:
        mean and variance
    """
    # get all the images
    images = []
    for link in input_directories:
        for extension in extensions:
            _images = glob.glob(os.path.join(link, "*." + extension))
            images += _images
            print("%s - %s: %d" % (link, extension, len(_images)))

    # Loop
    count = np.zeros(3)
    mean = np.zeros(3)
    M2 = np.zeros(3)
    for index, image_path in enumerate(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _count = image.shape[0] * image.shape[1]
        _values = np.sum(image, axis=(0, 1)) / _count / 255

        count += 1
        delta = _values - mean
        mean = mean + delta / count
        delta2 = _values - mean
        M2 = M2 + delta * delta2

    return mean, np.sqrt(M2 / (count - 1))


if __name__ == "__main__":
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", nargs='+',
                        default=["/home/VP/data/eo4belmap/solar_panels/data/data_set_netherlands_25_pngs/train/images"],
                        type=str, help="Path to a directories containing imagery data")
    parser.add_argument("--extensions", nargs='+',
                        default=["png", "jpg", "tif"],
                        type=str, help="List of image extensions to check")
    args = parser.parse_args()

    # Start main loop
    mean, variance = main(args.input_directory, args.extensions)
    print("Mean %s - std %s" % (mean, np.sqrt(variance)))
