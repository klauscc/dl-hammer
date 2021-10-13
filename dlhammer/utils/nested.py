# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================
import torch
from easydict import EasyDict


def map_structure(func, structure, args=(), kwargs={}):
    """apply func to each element in structure

    Args:
        func (callable): The function
        structure (dict, tuple, list): the structure to be mapped.

    Kwargs:
        args (tuple,list): The args to the function.
        kwargs (dict): The kwargs to the function.

    Returns: The same structure as input structure.

    """
    if structure is None:
        return None
    if isinstance(structure, (list, tuple)):
        return [func(x, *args, **kwargs) for x in structure]
    if isinstance(structure, dict):
        returns = dict()
        for key, value in structure.items():
            returns[key] = func(value, *args, **kwargs)
        return returns
    else:
        return func(structure, *args, **kwargs)


def nested_call(structure, func_name, args=(), kwargs={}):
    """call function for each element in nested structure.

    Args:
        structure (dict,tuple,list,object): If the structure is dict or list, then call func_name 
                                    for each values in the structure. 
                                    If structure is None, then the function will do nothing.
        func_name (string): function to call.

    Kwargs:
        args (tuple,list): The args to the called function.
        kwargs (dict): The kwargs to the function.

    """
    if structure is None:
        return
    if isinstance(structure, (list, tuple)):
        returns = []
        for element in structure:
            res = getattr(element, func_name)(*args, **kwargs)
            returns.append(res)
    elif isinstance(structure, dict):
        returns = EasyDict()
        for k, element in structure.items():
            returns[k] = getattr(element, func_name)(*args, **kwargs)
    else:
        returns = getattr(structure, func_name)(*args, **kwargs)
    return returns


def nested_to_device(data, device, non_blocking=True):
    """
        transfer data to device. The data is a nested structure
    Args:
        data (Dict, List, Tuple, torch.Tensor): nested data.
        device (torch.device): The target device.

    Kwargs: 
        non_blocking (Bool): Wether to block.

    Returns: The same structure of data. Each Tensor is transfered to `device`.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: nested_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [nested_to_device(v, device, non_blocking) for v in data]
    else:
        raise ValueError(f'Type {type(data)} is not supported.')
