import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from emmental.data import EmmentalDataset
from utils import convert_labels


class CXR8Dataset(EmmentalDataset):
    """Dataset to load NIH Chest X-ray 14 dataset

    Modified from reproduce-chexnet repo
    https://github.com/jrzech/reproduce-chexnet

    Args:
        name (string): name of dataset. Used by Emmental package.
        path_to_images (string): path to CXR8 images.
        path_to_labels (string): path to CXR8 labels.
        split (string): Fold of labels CSV to use.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        sample (int): Number of samples from labels CSV to use. sample==0 uses
            all samples in the dataset. Useful for testing.
        finding (string): Allows filtration by some finding. Use string "any" to
            use the entire dataset.
        seed (int): Random seed initialization for dataset.
        add_binary_triage_label (bool): adds a binary label for overall abnormality
            condition of a given sample.
    """

    def __init__(
        self,
        name,
        path_to_images,
        path_to_labels,
        split,
        transform=None,
        sample=0,
        finding="any",
        seed=0,
        add_binary_triage_label=False,
    ):
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.split = split
        self.df = pd.read_csv(self.path_to_labels)
        self.df = self.df[self.df["fold"] == split]
        self.seed = seed

        # can limit to sample, useful for testing
        # if split == "train" or split =="val": sample=500
        if sample > 0 and sample < len(self.df):
            self.df = self.df.sample(sample, random_state=self.seed)

        if (
            not finding == "any"
        ):  # can filter for positive findings of the kind described; for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print(
                        "No positive cases exist for "
                        + finding
                        + ", returning all unfiltered cases"
                    )
            else:
                print(
                    "cannot filter on finding "
                    + finding
                    + " as not in data - please check spelling"
                )

        self.df = self.df.set_index("Image Index")

        self.PRED_LABEL = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

        X_dict = {"image_name": []}
        Y_dict = {}

        if add_binary_triage_label:
            self.df["Abnormal"] = self.df[self.PRED_LABEL].sum(axis=1) > 0
            self.PRED_LABEL = self.PRED_LABEL + ["Abnormal"]

        for idx in range(len(self.df)):
            X_dict["image_name"].append(self.df.index[idx])
            for label in self.PRED_LABEL:
                if label not in Y_dict:
                    Y_dict[label] = []
                # +1 for 1 index
                Y_dict[label].append(self.df[label].iloc[idx].astype(int))

        for label in self.PRED_LABEL:
            Y_dict[label] = convert_labels(
                torch.from_numpy(
                    np.array(Y_dict[label])), "onezero", "categorical"
            )

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="image_name")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (x: dict, y: dict) where x is a dictionary mapping all
                possible inputs to EmmentalModel and y is a dictionary for all
                possible labels to EmmentalTasks
        """
        x_dict = {name: feature[index]
                  for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        image = Image.open(
            os.path.join(self.path_to_images, self.X_dict["image_name"][index])
        )
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        x_dict["image"] = image

        return x_dict, y_dict
