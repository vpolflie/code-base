"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
import torch

# Internal imports
from tools.exceptions import ConvertTypeException


def convert_to_numpy(x):
    """
    Convert a data type to a numpy array

    Args:
        x: A list, a tensorflow tensor, numpy array or a pytorch tensor

    Returns:
        A numpy array
    """
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], torch.Tensor):
            x = [convert_to_numpy(_x) for _x in x]
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ConvertTypeException(type(x))


def convert_to_pytorch(x, tensor_type=torch.FloatTensor):
    """
    Convert a data type to a pytorch tensor

    Args:
        x: A list, a tensorflow tensor, numpy array or a pytorch tensor
        tensor_type: a pytorch tensor type e.g. FloatTensor

    Returns:
        A pytorch tensor
    """
    if isinstance(x, list):
        return torch.stack([convert_to_pytorch(d) for d in x])
    elif isinstance(x, torch.Tensor):
        return x.type(tensor_type)
    #elif isinstance(x, tf.Tensor):
    #    return tensor_type(x.numpy())
    elif isinstance(x, np.ndarray):
        return tensor_type(x)
    else:
        raise ConvertTypeException(type(x))


def convert_to_serializable(x):
    """
    Converts data to the serializable version

    args:
        x: a tensorflow tensor, numpy array or a pytorch tensor

    returns:
        a serializable data object
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    elif isinstance(x, tf.Tensor):
        return x.numpy().tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return x
