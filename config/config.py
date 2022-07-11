"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External Imports
import os
from pathlib import Path
import uuid
import yaml

# Internal Imports
from tools.io.io_utils import read_yaml, load_config
from tools.path_utils import verify_path_exists
from tools.print_utils import red


def join(loader, node):
    """
    Function to handle joining paths in yaml file.

    When encountering '!join' tags, will treat subsequent items as
    a list of strings to be concatenated.

    Allows self-referencing paths like !join [\*BASE_PATH, /subdirectory/]

    Args:
        loader: yaml loader
        node: node in the yaml config which needs to be changed

    Returns:
        The sequence joined together into a single string
    """
    sequence = loader.construct_sequence(node)
    return ''.join([str(i) for i in sequence])


class Namespace:

    """
        Dummy class to define namespaces similar to the python "local" or "global" but then module independent.
        All given keyword arguments are transformed to attributes of this namespace.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _get_base_config(config, dict_key, base_key="base"):
    """
    Dictionary containing key value elements. This function checks if the current config has the base keyword, if so
    load in the new config and update the variables

    Args:
        config: dictionary python

    Parameters:
        base_key: The key identifier in the config where the value would contain a path to the base config

    Returns:
        An updated dictionary
    """
    if base_key in config:
        base_config = read_yaml(os.path.join("config", dict_key, config["base"] + ".yaml"))

        # Update base config
        for key, value in config.items():
            if base_key != key:
                base_config[key] = value
        config = _get_base_config(base_config, dict_key)
    return config


def update_module_config(dict_key, config):
    """
    Load a module specific config file and update the argument parser with the corresponding function

    Args:
        config: High level main config
        dict_key: The dictionary key for the current module that needs to be processed

    Returns:
        A dictionary containing module specific config and the updated parser
    """
    # Load in the corresponding config file and update the main configuration object
    _config = read_yaml(os.path.join("config", dict_key, config[dict_key] + ".yaml"))
    _config[dict_key + "_yaml"] = config[dict_key]

    # Get base config if available
    _config = _get_base_config(_config, dict_key)

    # To upper keys
    _config = dictionary_key_to_upper(_config)

    # Return the updated parser and config dictionary
    return _config


def dictionary_key_to_upper(dictionary):
    """
    Convert the key values of a dictionary to full upper case, this function is mainly used for the config dictionaries
    This so all the variables in upper case with in the code base can be easily recognised as config variables

    Args:
        dictionary: The python dictionary that needs to be processed

    Returns:
        A new python dictionary with only upper case key words
    """
    # Change all the config variables to upper case keys so they can easily be recognised in the code
    _dictionary = {}
    for k, v in dictionary.items():
        _dictionary[k.upper()] = v
    return _dictionary


def update_config(config_path="config.yaml"):
    """
    Config script, the local namespace of this script will contain all the different configuration of our models.

    Now since this is a code base multiple different models and data loaders will be present. These different
    implementations might not share the same variables or parameters. So this config object needs to be created in a
    modular way as well.

    The input for this function is a high level config file where most of the parameters are of a module-module_name
    pair. Each of these pairs will then be loaded in separately and the argument parser will be updated with specific
    arguments based on the chosen models.

    After all the arguments are loaded in correctly, a namespace is created for each module and can then be easily
    accessed by importing this python file and accessing the config parameter i.e.:

    import config.config as cfg
    cfg.MODULE.MODULE_VARIABLE

    Parameters:
        config_path: String containing the path to the main high level config file

    Returns:
        Nothing
    """

    # specify config yaml path default#
    config_path = Path(config_path)
    verify_path_exists(config_path)

    # register the tag handler
    yaml.SafeLoader.add_constructor(tag='!join', constructor=join)

    # load the yml file as a dict
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            raise e

    # Update the argument parser and load the yaml per module and create a namespace per module
    if "data_loader" in config and config["data_loader"] is not None:
        _config = update_module_config("data_loader", config)
        config["data_loader"] = Namespace(**_config)
    if "model" in config and config["model"] is not None:
        _config = update_module_config("model", config)
        config["model"] = Namespace(**_config)
    if "runner" in config and config["runner"] is not None:
        _config = update_module_config("runner", config)
        config["runner"] = Namespace(**_config)
    if "visualiser" in config and config["visualiser"] is not None:
        _config = update_module_config("visualiser", config)
        config["visualiser"] = Namespace(**_config)

    # Check if two modules have the same variables to avoid error when updating the config file
    dict_key_check = []
    for config_key, config_namespace in config.items():
        if isinstance(config_namespace, Namespace):
            for key in config_namespace.__dict__.keys():
                assert key not in dict_key_check, "There are duplicate variables in the config: %s" % key
                dict_key_check.append(key)

    # extra step - try to catch missing file paths early
    # make sure paths correspond to existing files, warning otherwise
    for config_key, config_namespace in config.items():
        if isinstance(config_namespace, Namespace):
            for key, value in config_namespace.__dict__.items():
                if 'PATH' in key and isinstance(value, str):
                    config[key] = Path(value)
                    verify_path_exists(config[key], raise_exc=False)

    # Add unique identifier
    if hasattr(config["runner"], "USE_HASH") and config["runner"].USE_HASH:
        config["runner"].OUTPUT = config["runner"].OUTPUT + "_" + uuid.uuid4().hex

    # Change all the config variables to upper case keys so they can easily be recognised in the code
    _config = dictionary_key_to_upper(config)

    # Check if we are loading in model and the config needs to be updated
    if _config["MODEL"].LOAD:
        if hasattr(_config["MODEL"], "UPDATE_CONFIG_LOADED") and _config["MODEL"].UPDATE_CONFIG_LOADED:
            _config["MODEL"] = load_config(os.path.join(_config["MODEL"].LOAD, "config.yaml"), _config["MODEL"])
        elif not hasattr(_config["MODEL"], "UPDATE_CONFIG_LOADED"):
            print(red("\nUpdating config to match loaded model, if this effect is "
                      "undesired please set the UPDATE_CONFIG_LOADED to False,"
                      "if it is update the config file to UPDATE_CONFIG_LOADED: True for verbosity"))
            _config["MODEL"] = load_config(os.path.join(_config["MODEL"].LOAD, "config.yaml"), _config["MODEL"])

    # update the local namespace with the values in the config dict, enabling
    # us to access them when importing this file as attributes `config.model` etc
    globals().update(_config)
