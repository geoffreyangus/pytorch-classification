from functools import partial

import torch.nn.functional as F
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn
from modules.torch_vision_encoder import TorchVisionEncoder


def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_ouput_dict[module_name][0]


def get_task(task_names):

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
                    f"{task_name}_pred_head": nn.Linear(
                        classification_layer_dim, 2
                    ),
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
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["accuracy", "roc_auc"]),
        )
        tasks.append(task)

    return tasks
