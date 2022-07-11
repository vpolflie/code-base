"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from tools.python_utils import Registry


def build(module, registry):
    """
        Create a visualiser to run a model

        Returns:
            a visualiser object
    """
    return registry.get(module)()


VISUALIZERS = Registry('visualizers', build)
