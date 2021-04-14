# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import os
import ast
import argparse
import datetime
from functools import partial
from ruamel.yaml import YAML
from easydict import EasyDict

# from .utils import get_vacant_gpu
from .logger import bootstrap_logger, log
from .utils.system import get_available_gpuids

args = EasyDict()


def bootstrap_args(default_params=None):
    """get the params from yaml file and args. The args will override arguemnts in the yaml file.
    Returns: EasyDict instance.

    """
    parser = define_default_arg_parser()
    params = update_arg_params(parser, default_params)
    args.update(params)

    create_workspace(args) #create workspace
    bootstrap_logger(get_logfile()) # setup logger
    setup_gpu(args.ngpu) #setup gpu
    log.info(params_string(args))

    return params


def setup_gpu(ngpu):
    gpuids = get_available_gpuids()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpuids[:ngpu]])
    log.info(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')


def get_logfile():
    if args.workspace == '':
        return args.logfile
    else:
        return os.path.join(args.workspace, args.logfile)


def params_string(params):
    """format params to a string

    Args:
        params (EasyDict): the params. 

    Returns: The string to display.

    """
    msg = "Params:\n\n"
    for k, v in params.items():
        msg += "{:20}:{}".format(k, v)
        msg += "\n"
    return msg


def define_default_arg_parser():
    """Define a default arg_parser.

    Returns: 
        A argparse.ArgumentParser. More arguments can be added.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="load configs from yaml file", default="", type=str)
    parser.add_argument("--workspace", help="Path to save the ckpts and results.", default="", type=str)
    parser.add_argument("--logfile", help="the filename of log", default=f"tfhammer.log", type=str)
    parser.add_argument("--ngpu", help="The number of gpu to use.", default=1, type=int)
    return parser


def update_arg_params(arg_parser, default_params=None):
    """ update argparser to args.

    Args:
        arg_parser: argparse.ArgumentParser.
    """

    parsed, unknown = arg_parser.parse_known_args()
    if default_params and parsed.yaml == "" and "yaml" in default_params:
        parsed.yaml = default_params["yaml"]

    params = EasyDict()
    params.update(default_params)

    # update params from yaml
    if os.path.isfile(parsed.yaml):
        yaml = YAML()
        yml_params = yaml.load(open(parsed.yaml, "r"))
        params.update(yml_params)

    # update default_params.
    if default_params:
        for k, v in default_params.items():
            if k not in params:
                params[k] = v

    # update arg_parser to params.
    for arg in unknown:
        if arg.startswith(("-", "--")):
            arg_parser.add_argument(arg)
    args = arg_parser.parse_args()
    dict_args = vars(args)

    for key, value in dict_args.items():    # override params from the arg_parser
        if key not in params:
            params[key] = value
        elif value != None and value != -1 and value != "":
            params[key] = value

    # eval str or keep.
    for k, v in params.items():
        params[k] = eval_string(v)

    return params


def create_workspace(params):
    """
    Args:
        params (EasyDict): The params.

    Returns:
        EasyDict. update the workspace.

    """
    # if workspace is not specified, do not create workspace.

    workspace = params.workspace
    if workspace != '':
        os.makedirs(workspace, exist_ok=True)
    return


def eval_string(string):
    """automatically evaluate string to corresponding types.
    
    For example:
        not a string  -> return the original input
        '0'  -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0,1,2]
        'eval(1+2)' -> 3
        'eval(range(5))' -> [0,1,2,3,4]


    Args:
        value : string.

    Returns: the corresponding type

    """
    if not isinstance(string, str):
        return string
    if len(string) > 1 and string[0] == '[' and string[-1] == ']':
        return eval(string)
    if string[0:5] == 'eval(':
        return eval(string[5:-1])
    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v
