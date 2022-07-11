"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from datetime import datetime
import re

# Internal imports


class PythonFont:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def bold(string):
    """
    Convert a string to a bold font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.BOLD + string + PythonFont.ENDC


def underline(string):
    """
    Convert a string to have a underline
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.UNDERLINE + string + PythonFont.ENDC


def purple(string):
    """
    Convert a string to have a purple font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.HEADER + string + PythonFont.ENDC


def blue(string):
    """
    Convert a string to have a blue font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.OKBLUE + string + PythonFont.ENDC


def cyan(string):
    """
    Convert a string to have a cyan font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.OKCYAN + string + PythonFont.ENDC


def green(string):
    """
    Convert a string to have a purple font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.OKGREEN + string + PythonFont.ENDC


def yellow(string):
    """
    Convert a string to have a purple font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.WARNING + string + PythonFont.ENDC


def red(string):
    """
    Convert a string to have a purple font
    Args:
        string: a string

    Returns:
        a string
    """
    return PythonFont.FAIL + string + PythonFont.ENDC


def add_time_stamp(string):
    """
    Add a time stamp to a string
    Args:
        string: a string

    Returns:
        a string
    """
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    timestamped_string = "[%s]    %s" % (timestamp, string)
    return timestamped_string


def progress_bar(length, progress):
    """
    String representation of a progress
    Args:
        length: length of the returned string
        progress: progress that needs to be represented (float)

    Returns:
        A string representing the progress in the following format: [=====>     ]
    """
    length -= 2
    if progress < 1.:
        progress = int(length * progress)
        negative_progress = length - progress
        negative_progress -= int(progress == 0)
        return "[" + "=" * (progress - 1) + ">" + " " * negative_progress + "]"
    else:
        return "[" + "=" * length + "]"


def byte_strip(string):
    """
    Strip bytes from a string
    Args:
        string: a string

    Returns:
        a stripped string
    """
    for k, v in PythonFont.__dict__.items():
        if isinstance(v, str):
            string = string.replace(v, "")
    return string
