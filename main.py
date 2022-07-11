"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import argparse
import rasterio

# Internal imports
import config.config as config
from data_loader.builder import DATA_LOADERS
import logger.logger as log
from model.builder import MODELS
from runner.builder import RUNNERS
from visualiser.builder import VISUALIZERS


def run_framework(config_file_path):
    """
    This function executes the pipeline of our code base. This pipeline can be defined on a high level by a config file.
    Further specification to the config of the pipeline can be made using module specific config files or using command
    line arguments which will then overwrite any previously defined values.

    A normal run of our pipeline will be:

    config -> data loader -> model -> runner
                                       ^ |
                                       | v
                                    visualiser

    Each of these modules will share a common interface and should allow developers to easily switch different blocks
    in and out.

    Args:
        config_file_path: path to the main config file

    Returns:
        Nothing
    """
    # Save up 5 GB of RAM for rasterio
    with rasterio.Env(GDAL_CACHEMAX=5000000000) as env:
        config.update_config(config_file_path)
        log.logger.initialise()
        data_loader = DATA_LOADERS.build(config.DATA_LOADER.DATA_LOADER_MODULE)
        model = MODELS.build(config.MODEL.MODEL_MODULE)
        visualiser = VISUALIZERS.build(config.VISUALISER.VISUALISER_MODULE)
        runner = RUNNERS.build(config.RUNNER.RUNNER_MODULE, data_loader, model, visualiser)
        runner.run()


if __name__ == "__main__":
    # Initialise argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        default="./config.yaml",
                        type=str,
                        help="Path to the main config file")
    args = parser.parse_args()

    # Start main loop
    run_framework(args.config_file)
