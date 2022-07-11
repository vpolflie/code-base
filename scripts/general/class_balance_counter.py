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


def main(input_directory):
    """
    Loops over segmentations and counts the values, at the end the total count of each class is returned

    Args:
        input_directory: path to the input directory

    Returns:
        A list of unique values, a list containing the corresponding counts
    """
    # Loop over
    total_count = {}
    unique = None

    if isinstance(input_directory, list):
        segmentations = []
        for input_dir in input_directory:
            segmentations += glob.glob(os.path.join(input_dir, "*.png"))
    else:
        segmentations = glob.glob(os.path.join(input_directory, "*.png"))

    for index, segmentation_path in enumerate(segmentations):
        print(segmentation_path)
        _segmentation = cv2.imread(segmentation_path)[..., 0:1]

        unique, count = np.unique(_segmentation, return_counts=True)

        for unique_index, unique_value in enumerate(unique):
            if unique_value in total_count:
                total_count[unique_value] += count[unique_index]
            else:
                total_count[unique_value] = count[unique_index]

    return total_count


if __name__ == "__main__":
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", default="../../data/gardens_drive/data_set_winter_filtered_segm/train/segmentations",
                        nargs="+", type=str,
                        help="Path to a directory structure of a data set")
    args = parser.parse_args()

    # Start main loop
    total_counts = main(args.input_directory)

    for unique_value, count in total_counts.items():
        print("%d: \t\t %d" % (unique_value, count))
