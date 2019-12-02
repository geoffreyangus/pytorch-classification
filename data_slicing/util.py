import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from emmental import Meta
import transforms as custom_transforms

def compose(fn_list):
    """
    Compiles a list of functions described as {class_name: str, args: str}.
    """
    transforms_list = []
    for f in fn_list:
        class_name = f['class_name']
        args = f['args']
        if hasattr(transforms, class_name):
            transforms_list.append(getattr(transforms, class_name)(**args))
        else:
            transforms_list.append(getattr(custom_transforms, class_name)(**args))
    return transforms.Compose(transforms_list)


def ce_loss(task_name, immediate_output, Y, active):
    """
    CrossEntropyLoss function to be used with Emmental module.
    """
    return F.cross_entropy(
        immediate_output[f"decoder_module_{task_name}"][0][active], (Y.view(-1) - 1)[active]
    )


def output(task_name, immediate_output):
    """
    Softmax function to be used with Emmental module.
    """
    return immediate_output[f"decoder_module_{task_name}"][0]


def require_dir(dir_str):
    """
    """
    if not(osp.exists(dir_str)):
        require_dir(osp.dirname(dir_str))
        os.mkdir(dir_str)


def get_sample_weights(dataset, weight_task, class_probs):
    """
    """
    classes = dataset.Y_dict[weight_task]
    classes = torch.LongTensor(classes)
    counts = torch.bincount(classes)

    weights = torch.zeros_like(classes, dtype=torch.float)
    for example_idx, class_idx in enumerate(classes):
        class_prob = class_probs[class_idx] / float(counts[class_idx])
        weights[example_idx] = class_prob
    return weights


def convert_labels(Y, source, target):
    """Convert a matrix from one label type to another

    Args:
        Y: A np.ndarray or torch.Tensor of labels (ints) using source convention
        source: The convention the labels are currently expressed in
        target: The convention to convert the labels to
    Returns:
        Y: an np.ndarray or torch.Tensor of labels (ints) using the target convention

    Conventions:
        'categorical': [0: abstain, 1: positive, 2: negative]
        'plusminus': [0: abstain, 1: positive, -1: negative]
        'onezero': [0: negative, 1: positive]

    Note that converting to 'onezero' will combine abstain and negative labels.
    """
    if Y is None:
        return Y
    if isinstance(Y, np.ndarray):
        Y = Y.copy()
        assert Y.dtype == np.int64
    elif isinstance(Y, torch.Tensor):
        Y = Y.clone()
        assert np.sum(Y.cpu().numpy() - Y.cpu().numpy().astype(int)) == 0.0
    else:
        raise ValueError("Unrecognized label data type.")
    negative_map = {"categorical": 2, "plusminus": -1, "onezero": 0}
    Y[Y == negative_map[source]] = negative_map[target]
    return Y


def write_to_file(file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(osp.join(Meta.log_path, file_name), "w")
    fout.write(value + "\n")
    fout.close()
