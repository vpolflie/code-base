"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports

# Internal imports
from tools.python_utils import Registry


def build(module, data_loader, model, visualiser, registry=None):
    """
    Build a runner object
    Args:
        module: module to build
        data_loader: data loader object
        model: a model object
        visualiser: a visualiser object

    Parameters:
        registry: where the model is registered

    Returns:
        A runner object
    """
    # Initialise the runner object based on the runner module of the config
    runner = registry.get(module)(data_loader, model, visualiser)
    runner.initialise()
    return runner


RUNNERS = Registry('runners', build)
