"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import math
import os
import sys

# Internal imports
import config.config as cfg
from config.config import Namespace
from tools.path_utils import create_folders
from tools.print_utils import *


""""
Logger module for the ai code base
"""


class Logger:

    """
        Custom logger for the ai code base
    """

    def __init__(self):
        # Initialise logger file variable
        self.log_file = None

        # Initialise variables
        self.phase = None

    def initialise(self):
        """
        Initialise the logger
        Returns:

        """
        # Initialise logger file
        log_path = os.path.join("meta_data", "models", cfg.RUNNER.OUTPUT)
        create_folders(log_path)
        self.log_file = open(os.path.join(log_path, "log.txt"), "w")

        # Print config summary
        self._log_config_summary()

    def _trigger_phase_change(self, phase):
        """
        Change the current phase
        Args:
            phase: the new phase in string format

        Returns:

        """
        # Process phase change
        self.phase = phase.upper()

        # Log phase change
        barrier = "##############################################"
        self.log("")
        self.log("")
        self.log(yellow(barrier))
        spacing_length = len(barrier) - 2 - len(phase)
        self.log(yellow("#" + math.ceil(spacing_length / 2) * " " + phase + " " * math.floor(spacing_length / 2) + "#"))
        self.log(yellow(barrier))
        self.log("")
        self.log("")

    def log(self, string, phase="", start="", end="\n"):
        """
        Log a message to the std and the log file

        Args:
            string: a string to print

        Parameters:
            phase: a string representing the phase
            start: start of the line when writing to std
            end: end of the line when writing to std

        Returns:

        """
        # Check phase
        if phase and self.phase != phase.upper():
            self._trigger_phase_change(phase)

        # Log string
        self.log_file.write(add_time_stamp(byte_strip(string + "\n")))
        sys.stdout.write(start + string + end)
        sys.stdout.flush()

    def _log_config_summary(self):
        """
        Log a summary of the config
        Returns:

        """
        # Go over all the items in config and only extract the Namespace Objects
        for k, v in sorted(cfg.__dict__.items()):
            if isinstance(v, Namespace):
                for attribute, value in sorted(v.__dict__.items()):
                    # Create print string
                    string = "%s:" + " " * (40 - 1 - len(attribute)) + "%s"

                    # If it is a module extract the name of the module and print this instead
                    if "MODULE" in attribute:
                        attribute = cyan(bold(attribute))
                    else:
                        attribute = blue(attribute)

                    # Log
                    self.log(string % (attribute, str(value)), "CONFIG")

                self.log("", "CONFIG")


logger = Logger()
