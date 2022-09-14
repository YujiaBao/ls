import os
import yaml

from .print import print


def _recursive_update(origin_dict: dict, new_dict: dict):
    '''
        Update the values in the origin_dict based on the new_dict.
        If the value is a dictionary, we will apply this approach recursively.
    '''
    for k, v in new_dict.items():
        if type(v) is dict:
            if k not in origin_dict:
                origin_dict[k] = v
            else:
                origin_dict[k] = _recursive_update(origin_dict[k], v)
        else:
            origin_dict[k] = v

    return origin_dict


def read_config(overwrite_config: dict):
    '''
        Load the configuration file for ls.
    '''
    # Load the default general configuration file
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    ls_dir = '/'.join(utils_dir.split('/')[:-1])
    default_cfg_path = os.path.join(ls_dir, 'configs/default.yaml')

    with open(default_cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration based on the overwrite parameters
    _recursive_update(cfg, overwrite_config)

    return cfg
