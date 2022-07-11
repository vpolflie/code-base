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
import re

# Internal imports
from tools.path_utils import create_folders, symbolic_link_force


def main(input_directory, output_directory):
    """
    Create a structured data set using symbolic link. Data set will be of the ai-code-base defined coco structure

    Args:
        input_directory: path to the input directory
        output_directory: path to the output directory

    Returns:
        Nothing
    """
    # Create the folders
    for split in ["train", "validation", "test"]:
        for data_type in ["images", "segmentations"]:
            create_folders(os.path.join(output_directory, split, data_type))

    # Get the images
    images = glob.glob(os.path.join(input_directory, "image", "*.png"))

    # Loop over
    for index, image_path in enumerate(images):
        segmentation_path = re.sub("image", "segmentation", image_path)

        # Do validation/test specifications
        if False:
            split = "validation"
        else:
            split = "train"

        symbolic_link_force(
            os.path.abspath(image_path), os.path.join(output_directory, split, "images", "%08d.png" % index)
        )
        symbolic_link_force(
            os.path.abspath(segmentation_path), os.path.join(output_directory, split, "segmentations", "%08d.png" % index)
        )


if __name__ == "__main__":
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", default="data/solar_panels_local/netherlands_6_5_pngs/*/", type=str,
                        help="Path to a directory structure of a data set")
    parser.add_argument("--output_directory", default="data/solar_panels_local/data_set_netherlands_6_5_pngs", type=str,
                        help="Path to the preferred output directory")
    args = parser.parse_args()

    # Start main loop
    main(args.input_directory, args.output_directory)
