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
from pathlib import Path

# Internal imports
from tools.data.shape import SEGMENTATION_CHANNELS_IMAGE_CODING as color_coding


def main(input_directory, extensions, number_of_categories, output_directory):
    """
    Function which calculates an approximation of the mean and variance given a series of input directories containing
    images,

    Args:
        input_directory: a list of paths to image directories
        extensions: a list of image extensions
        number_of_categories: number of categories that need to be converted
        output_directory: the directory where the new images need to be saved

    Returns:
        mean and variance
    """
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # get all the images
    segmentations = []
    for extension in extensions:
        segmentations += glob.glob(os.path.join(input_directory, "*." + extension))

    # Convert segmentations
    for index, segmentation_path in enumerate(segmentations):
        segmentation = cv2.imread(segmentation_path)[..., 0]
        file_name = segmentation_path.split("/")[-1]

        image = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        for i in range(number_of_categories):
            image[segmentation == i] = color_coding[i]

        cv2.imwrite(os.path.join(output_directory, file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory",
                        default="/home/VP/data/eo4belmap/solar_panels/data/data_set_netherlands_25_pngs/train/images",
                        type=str, help="Path to a directories containing imagery data")
    parser.add_argument("--extensions", nargs='+',
                        default=["png", "jpg", "tif"],
                        type=str, help="List of image extensions to check")
    parser.add_argument("--number-of-categories", default=4,
                        type=int, help="Number of categories to convert")
    parser.add_argument("--output-directory", default="/home/VP/data/eo4belmap/solar_panels/data/data_set_netherlands_25_pngs/train/images",
                        type=str, help="Path to a directories containing imagery data")

    args = parser.parse_args()

    # Start main loop
    main(args.input_directory, args.extensions, args.number_of_categories, args.output_directory)
