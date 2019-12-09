import sys
import os
import os.path as osp

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
from emmental import Meta

import transforms as custom_transforms

def ce_loss(task_name, immediate_output_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_output_dict[module_name][0][active], Y.view(-1)[active]
    )


def output(task_name, immediate_output_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_output_dict[module_name][0]


def pair_iic_loss(task_name, immediate_output_dict, Y, active,
                  lamb=1.0, EPS=sys.float_info.epsilon):
    module_name = f"{task_name}_pred_head"
    x_out, x_tf_out = [F.softmax(x, dim=-1)
                       for x in immediate_output_dict[module_name]]

    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    loss = loss.sum()

#     loss_no_lamb = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
#     loss_no_lamb = loss_no_lamb.sum()

    return loss  # , loss_no_lamb


def pair_output(task_name, immediate_output_dict):
    module_name = f"{task_name}_pred_head"
    return torch.stack(immediate_output_dict[module_name], dim=2)


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def compose(fn_list):
    """
    Compiles a list of functions described as {class_name: str, args: str}.
    """
    if fn_list:
        transforms_list = []
        for f in fn_list:
            class_name = f['class_name']
            args = f['args']
            if hasattr(transforms, class_name):
                transforms_list.append(getattr(transforms, class_name)(**args))
            else:
                transforms_list.append(getattr(custom_transforms, class_name)(**args))
        return transforms.Compose(transforms_list)
    else:
        return None
    

def write_to_file(file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(osp.join(Meta.log_path, file_name), "w")
    fout.write(value + "\n")
    fout.close()
