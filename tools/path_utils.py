# External Imports
import errno
import logging
import os
from pathlib import Path
from shutil import rmtree

log = logging.getLogger(__name__)


def verify_path_exists(path, raise_exc=True):
    """
    Check a path exists and raise a FileNotFoundError if not
    Args:
        path: String or Path object containing a path to a file

    Parameters:
        raise_exc: Boolean indicating whether to raise an exception

    Returns:

    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        message = f"File {path} not found!"
        if raise_exc:
            raise FileNotFoundError(message)
        # warn
        log.warning(message)


def create_folders(path, remove=False):
    """
    Create a nested folder structure and remove the previous folders if they already existed
    Args:
        path: path of folders to create

    Parameters:
        remove: boolean whether to remove the previous folder ror not

    Returns:

    """
    # Get the path
    folder_path = Path(path)

    # If remove and the path already exists delete
    if remove and folder_path.exists():
        rmtree(folder_path)

    # Create the folder
    folder_path.mkdir(parents=True, exist_ok=not remove)


def symbolic_link_force(target, link_name):
    """
    Force create a symbolic link, this will overwrite the previous symbolic link

    Args:
        target: target path to create a symbolic link to
        link_name: path of the symbolic link

    Returns:

    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
