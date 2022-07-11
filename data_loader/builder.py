"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import os

# Internal imports
from data_loader.data_set_configuration import DataSetConfiguration
from tools.io.io_utils import read_yaml
from tools.path_utils import verify_path_exists
from tools.print_utils import *
from tools.python_utils import Registry

import config.config as cfg
import logger.logger as log
logger = log.logger


def build(module, registry):
    """
    This function first builds the specified data configurations and then uses these to build a data loader based
    on the config parameters.

    Returns:
        A data loader object

    """
    # Collect all the required data sets
    data_sets_names = []
    if cfg.RUNNER.TRAIN:
        data_sets_names += cfg.DATA_LOADER.DATA_SETS_TRAIN
        if cfg.RUNNER.VALIDATION:
            data_sets_names += cfg.DATA_LOADER.DATA_SETS_VALIDATION
    else:
        data_sets_names += cfg.DATA_LOADER.DATA_SETS_TEST

    # Loop over all mentioned data sets and create data set object for each of them
    data_sets = {}
    for data_set_name in set(data_sets_names):
        # Check if a config file exists for each data set
        data_set_config_path = os.path.join("config", "data_set_configuration", data_set_name + ".yaml")
        verify_path_exists(data_set_config_path)

        # Read yaml and convert keys
        data_set_config = read_yaml(data_set_config_path)

        # Create data set
        data_sets[data_set_name] = DataSetConfiguration(data_set_config, data_set_name)

    # Log information
    for data_set_name, data_set_configuration in data_sets.items():
        logger.log(bold(data_set_name), phase="DATA SETS")
        logger.log("", phase="DATA SETS")
        logger.log(underline("Train files:"), phase="DATA SETS")
        for k, v in data_set_configuration.train_file_paths.items():
            logger.log("%s: %d" % (k, len(v)), "DATA SETS")
        logger.log("", phase="DATA SETS")

        logger.log(underline("Validation files:"), phase="DATA SETS")
        for k, v in data_set_configuration.validation_file_paths.items():
            logger.log("%s: %d" % (k, len(v)), phase="DATA SETS")
        logger.log("", phase="DATA SETS")

        logger.log(underline("Test files:"), phase="DATA SETS")
        for k, v in data_set_configuration.test_file_paths.items():
            logger.log("%s: %d" % (k, len(v)), phase="DATA SETS")

        logger.log("", phase="DATA SETS")
        logger.log(underline("Additional Information"), phase="DATA SETS")
        logger.log(data_set_configuration.additional_information, phase="DATA SETS")

        logger.log("", phase="DATA SETS")
        logger.log("", phase="DATA SETS")

    # Build the correct data loader
    return registry.get(module)(data_sets)


DATA_LOADERS = Registry('data_loaders', build)
