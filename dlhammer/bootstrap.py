# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

from .logger import bootstrap_logger, logger
from .argparser import bootstrap_args, get_logfile
from .utils.misc import to_string

LOGGER_SET_FLAG = False

def bootstrap(default_cfg=None, print_cfg=True):
    """TODO: Docstring for bootstrap.

    Kwargs:
        use_argparser (TODO): TODO
        use_logger (TODO): TODO

    Returns: TODO

    """
    config = bootstrap_args(default_cfg)
    logger = setup_logger(config)
    if print_cfg:
        logger.info(to_string(config))
    return config


def setup_logger(cfg):
    """setup logger

    Args:
        cfg (dict): The log file will be 'cfg.WORKSPACE/cfg.LOG_NAME'

    Returns: logger

    """
    global LOGGER_SET_FLAG
    global logger
    if not LOGGER_SET_FLAG:
        logger = bootstrap_logger(get_logfile(cfg))
        LOGGER_SET_FLAG = True
    else:
        return logger


config = bootstrap(print_cfg=False)
logger = setup_logger(config)
