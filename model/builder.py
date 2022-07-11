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
        Create a model object that will output information to the commandline
    """
    model = registry.get(module)()
    model.build()
    model.initialize_model()
    return model


MODELS = Registry('models', build)
