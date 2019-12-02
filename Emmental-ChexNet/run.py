import argparse
import logging
import os
import sys

import emmental
import pandas as pd
from emmental import Meta
from emmental.contrib import slicing
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_arg import parse_arg, parse_arg_to_config
from emmental.utils.utils import str2bool
from torchvision import transforms

from dataset import CXR8Dataset
from slicing_functions import slicing_function_dict
from task import get_task
from task_config import CXR8_TASK_NAMES
from transforms import get_data_transforms

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

    parser.add_argument(
        "--max_data_samples", type=int, default=0, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--slices", type=str2bool, default=False, help="Whether to include slices"
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "ChexNet Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_arg(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_arg_to_config(args)

    emmental.init(config["meta_config"]["log_path"], config=config)

    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file("cmd.txt", cmd_msg)

    logger.info(f"Config: {Meta.config}")
    write_to_file("config.txt", Meta.config)

    DATA_NAME = "CXR8"

    task_names = CXR8_TASK_NAMES

    # Load dataset
    task_to_label_dict = {task_name: task_name for task_name in CXR8_TASK_NAMES}

    cxr8_transform = get_data_transforms(DATA_NAME)

    dataloaders = []
    tasks = []

    for split in ["train", "val", "test"]:
        dataset = CXR8Dataset(
            name=DATA_NAME,
            path_to_images=args.image_path,
            path_to_labels=args.data_path,
            split=split,
            transform=cxr8_transform[split],
            sample=args.max_data_samples,
            seed=1701,
        )
        logger.info(f"Loaded {split} split for {DATA_NAME}.")

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size,
                num_workers=8,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")

    standard_tasks = get_task(task_names)

    for task in standard_tasks:
        if args.slices and task.name in slicing_function_dict:
            slicing.add_slice_labels(task, dataloaders, slicing_function_dict[task.name])
            slice_tasks = slicing.build_slice_tasks(task, slicing_function_dict[task.name])
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
