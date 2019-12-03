import os
import functools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from torch.autograd import Variable


class LinearDecoder(nn.Module):

    def __init__(self, num_classes, num_layers=None, encoding_size=None, dropout_p=0.0):
        """
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features=encoding_size,
                                         out_features=encoding_size))
            self.layers.append(nn.Dropout(p=dropout_p))
        # classification layer
        self.layers.append(nn.Linear(in_features=encoding_size,
                                     out_features=num_classes))

    def forward(self, x):
        """
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ClippedDenseNet(nn.Module):

    def __init__(self, n_classes=14, pretrained=True, weights_path=None):
        """
        """
        super().__init__()
        self.densenet121 = densenet121(pretrained=pretrained)
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(self.densenet121.classifier.in_features, n_classes),
            nn.Sigmoid()
        )

        if weights_path:
            state_dict = torch.load(weights_path)['state_dict']
            state_dict = {k.replace('module.densenet121', 'densenet121'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.norm.1', '.norm1'): v for k, v  in state_dict.items()}
            state_dict = {k.replace('.norm.2', '.norm2'): v for k, v  in state_dict.items()}
            state_dict = {k.replace('.conv.1', '.conv1'): v for k, v  in state_dict.items()}
            state_dict = {k.replace('.conv.2', '.conv2'): v for k, v  in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
            num_loaded = len(set(self.state_dict().keys()) & set(state_dict.keys()))
            num_total = len(state_dict.keys())
            if num_loaded < num_total:
                missing_params = set(state_dict.keys()).symmetric_difference(set(self.state_dict().keys()))
                logging.info("Could not load these parameters due to name mismatch: " + str(missing_params))
            logging.info(f"Loaded {num_loaded}/{num_total} pretrained parameters")

        self.densenet121.classifier = nn.Identity()

    def forward(self, x):
        """
        """
        return self.densenet121(x)