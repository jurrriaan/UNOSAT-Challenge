
import gc
import os
from datetime import datetime


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
