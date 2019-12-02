import os
import os.path as osp

import transforms as custom_transforms
import torchvision.transforms as transforms
import torch.nn.functional as F


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
        immediate_output[f"decoder_module_{task_name}"][0][active], (Y.view(-1))[active]
    )


def output(task_name, immediate_output):
    """
    Softmax function to be used with Emmental module.
    """
    return F.softmax(immediate_output[f"decoder_module_{task_name}"][0], dim=1)


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
