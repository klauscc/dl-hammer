# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import sys
import logging

from .logger import bootstrap_logger, log
from .argparser import bootstrap_args, args

__all__ = ['bootstrap', 'log', 'args']


def bootstrap(default_args=None):
    """TODO: Docstring for bootstrap.

    Kwargs:
        use_argparser (TODO): TODO
        use_logger (TODO): TODO

    Returns: TODO

    """
    args = bootstrap_args(default_args)
    return args, log
