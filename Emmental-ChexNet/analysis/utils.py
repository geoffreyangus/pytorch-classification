import json
import os

import numpy as np
import pandas as pd


def load_log_file(log_file):
    """
    Loads Emmental txt-formatted log file
    """
    with open(log_file, "r") as data_file:
        reader = data_file.read()
        data = eval(reader)
    return data


def load_results_from_log(log_dir):
    """
    Load all json logs from Emmental log dict
    """
    results = {}
    log_files = [a for a in os.listdir(log_dir) if a.endswith("txt") and "cmd" not in a]
    for fl in log_files:
        path = os.path.join(log_dir, fl)
        fl_str = fl.split(".")[0]
        results[fl_str] = load_log_file(path)
    return results


def get_metric_attrs(nm):
    tsk, met = nm.split(":")
    slc, dataset, split, metric = met.split("/")
    return tsk, slc, dataset, split, metric

    Abnormal: Atelectasis / CXR8 / train / accuracy


def get_task_name(nm):
    nm_red = "_".join(nm.split("_")[1:]).split(":")[0]
    return nm_red


def get_labelset_name(nm):
    return "_".join(nm.split("_")[1:])
