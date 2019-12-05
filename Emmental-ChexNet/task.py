from functools import partial

import torch.nn.functional as F
from torch import nn

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.torch_vision_encoder import TorchVisionEncoder


def weighted_ce_loss(task_name, class_weights, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active],
        weight=class_weights
    )


def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_ouput_dict[module_name][0]


def get_task(task_names, task_to_class_weights):

    CNN_ENCODER = "densenet121"

    input_shape = (3, 224, 224)

    cnn_module = TorchVisionEncoder(CNN_ENCODER, pretrained=True)
    classification_layer_dim = cnn_module.get_frm_output_size(input_shape)

    tasks = []

    for task_name in task_names:
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    f"cnn": cnn_module,
                    f"{task_name}_pred_head": nn.Linear(classification_layer_dim, 2),
                }
            ),
            task_flow=[
                {"name": "feature", "module": "cnn", "inputs": [("_input_", "image")]},
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("feature", 0)],
                },
            ],
            loss_func=partial(weighted_ce_loss, task_name, task_to_class_weights[task_name]),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["accuracy", "f1", "roc_auc"]),
        )
        tasks.append(task)
    return tasks
