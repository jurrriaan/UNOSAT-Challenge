
import gc
import numpy as np
import os
from datetime import datetime
import yaml

def make_dir(path_dir):
    if not os.path.exists(path_dir):
       os.mkdir(path_dir)
    return path_dir


def datetime_now(fmt_datetime="%Y%m%d_%H%M%S"):
    return datetime.strftime(datetime.now(), fmt_datetime)


def remove_from_RAM(*args):
    for arg in args:
        del arg
    gc.collect()


def convert2integers(arr, dtype):
    n_max = get_nmax_dtype(dtype)
    return np.array(n_max * arr, dtype=dtype)


def get_nmax_dtype(dtype):
    if dtype == np.uint8:
        dtype_num = 8
        n_max = 2 ** dtype_num
    elif dtype == np.uint16:
        dtype_num = 16
        n_max = 2 ** dtype_num
    elif dtype == np.uint32:
        dtype_num = 32
        n_max = 2 ** dtype_num
    else:
        n_max = 1
    return n_max


def save_yaml_parms(path_save, dict_parms, fn_yaml):
    with open(os.path.join(path_save, fn_yaml), 'w') as file:
        yaml.dump(dict_parms, file)
