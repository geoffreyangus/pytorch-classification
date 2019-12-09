import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def score_slices(model, dataloaders, task_names, slice_func_dict):
    assert isinstance(dataloaders, list)
    assert isinstance(task_names, list)
    assert isinstance(slice_func_dict, dict)
    scores = {}
    for task_name in task_names:
        scorer = model.scorers[task_name]
        for dataloader in dataloaders:
            logging.info(f"Evaluating on task {task_name}, {dataloader.split} split")
            pred_dict = model.predict(dataloader, return_preds=True)
            golds = np.array(pred_dict["golds"][task_name])
            probs = np.array(pred_dict["probs"][task_name])
            preds = np.array(pred_dict["preds"][task_name])
            split_scores = scorer.score(golds, probs, preds)
            scores.update(split_scores)
            for slice_name, slice_func in slice_func_dict.items():
                logging.info(f"Evaluating slice {slice_name}")
                inds = slice_func(dataloader.dataset)
                mask = (inds == 1).numpy().astype(bool)
                slice_scores = scorer.score(golds[mask], probs[mask], preds[mask])
                for metric_name, metric_value in slice_scores.items():
                    identifier = "/".join(
                        [
                            f"{task_name}_{slice_name}",
                            dataloader.data_name,
                            dataloader.split,
                            metric_name,
                        ]
                    )
                    scores[identifier] = metric_value
    return scores


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
