import argparse
import logging
import os
import sys

import torch.nn as nn
import torch

import emmental
from dataset import CXR8Dataset
from emmental import Meta
from emmental.contrib import slicing
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import str2bool, move_to_device
from slicing_functions import slicing_function_dict, slicing_function_eval_dict
from task import get_task
from task_config import CXR8_TASK_NAMES
from transforms import get_data_transforms
from utils import score_slices

logger = logging.getLogger(__name__)


def write_to_file(file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(Meta.log_path, file_name), "w")
    fout.write(value + "\n")
    fout.close()


def add_application_args(parser):

    parser.add_argument("--data_path", type=str, help="The path to csv file")

    parser.add_argument("--image_path", type=str, help="The path to image files")

    parser.add_argument("--batch_size", type=int, default=5, help="batch size")

    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")

    parser.add_argument(
        "--max_data_samples", type=int, default=0, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--slices", type=str2bool, default=False, help="Whether to include slices"
    )
    parser.add_argument(
        "--tasks",
        default="CXR8",
        type=str,
        nargs="+",
        help="list of tasks; if CXR8, all CXR8; if TRIAGE, normal/abnormal",
    )
    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


def get_parser():
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "ChexNet Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)
    return parser


def main(args):
 
    # Ensure that global state is fresh
    Meta.reset()

    # Add args to config
    config = parse_args_to_config(args)

    # Initialize Emmental
    emmental.init(config["meta_config"]["log_path"], config=config)

    # Print command line expression to file
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file("cmd.txt", cmd_msg)

    # Print config to file
    logger.info(f"Config: {Meta.config}")
    write_to_file("config.txt", Meta.config)

    # Define data and task names
    DATA_NAME = "CXR8"
    task_names = CXR8_TASK_NAMES

    # Getting task to label dict
    # All possible tasks in dataloader
    all_tasks = CXR8_TASK_NAMES + ["Abnormal"]
    if "CXR8" in args.tasks:
        # Standard chexnet
        logging.info("Using all CXR8 tasks")
        task_names = CXR8_TASK_NAMES
        add_binary_triage_label = False
    elif "TRIAGE" in args.tasks:
        # Binary triage
        logging.info("Using only Abnormal task")
        task_names = ["Abnormal"]
        add_binary_triage_label = True
    else:
        # Otherwise, making sure tasks are valid
        logging.info("Using only specified tasks")
        task_names = args.tasks
        for task in task_names:
            assert task in all_tasks
        add_binary_triage_label = True

    # Load dataset
    task_to_label_dict = {task_name: task_name for task_name in task_names}
    cxr8_transform = get_data_transforms(DATA_NAME)

    dataloaders = []
    tasks = []

    task_to_class_weights = {}
    for split in ["train", "val", "test"]:
        dataset = CXR8Dataset(
            name=DATA_NAME,
            path_to_images=args.image_path,
            path_to_labels=args.data_path,
            split=split,
            transform=cxr8_transform[split],
            sample=args.max_data_samples,
            seed=1701,
            add_binary_triage_label=add_binary_triage_label,
        )
        logger.info(f"Loaded {split} split for {DATA_NAME}.")

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size,
                num_workers=16,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")
        
        if split == 'train':
            for task_name in task_names:
                task_labels = dataset.Y_dict[task_to_label_dict[task_name]]
                # weighting scheme from paper: w_pos = |N| / (|P| + |N|), w_neg = |P| / (|P| + |N|)
                w_pos = sum(task_labels == 2).type(torch.FloatTensor) / len(task_labels) # categorical: [0: abstain, 1: positive, 2: negative]
                w_neg = sum(task_labels == 1).type(torch.FloatTensor) / len(task_labels)
                task_to_class_weights[task_name] = move_to_device(torch.tensor([w_pos, w_neg]), Meta.config["model_config"]["device"])
    standard_tasks = get_task(task_names, task_to_class_weights)

    # Slice indicator head module
    #slice_ind_head_module = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 2))
    slice_ind_head_module = None

    for task in standard_tasks:
        if args.slices and task.name in slicing_function_dict:
            slice_distribution = slicing.add_slice_labels(
                task, dataloaders, slicing_function_dict[task.name]
            )
            slice_tasks = slicing.build_slice_tasks(
                task,
                slicing_function_dict[task.name],
                slice_distribution,
                dropout=args.dropout,
                slice_ind_head_module=slice_ind_head_module,
            )
            tasks.extend(slice_tasks)
        else:
            tasks.append(task)

    # Build Emmental model
    model = EmmentalModel(name=DATA_NAME, tasks=tasks)

    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])
    # Training
    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # If model is slice-aware, slice scores will be calculated from slice heads
    # If model is not slice-aware, manually calculate performance on slices
    slice_func_dict = {}
    slice_keys = task_names

    for k in slice_keys:
        slice_func_dict.update(slicing_function_eval_dict[k])

    if len(slice_func_dict) > 0:
        scores = score_slices(model, dataloaders, task_names, slice_func_dict)
    else:
        scores = model.score(dataloaders)

    # Save metrics into file
    logger.info(f"Metrics: {scores}")
    write_to_file("metrics.txt", scores)

    # Save best metrics into file
    if args.train:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            "best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )

    logger.info(f"Logs written to {Meta.log_path}")


if __name__ == "__main__":
    # Getting parser and updating config
    parser = get_parser()
    args = parser.parse_args()
    main(args)
