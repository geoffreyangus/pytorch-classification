import os
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import densenet121

from torch.nn import Linear

logger = logging.getLogger(__name__)

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

        # TODO: get this to work
        if weights_path:
            state_dict = torch.load(weights_path)['state_dict']
            state_dict = {k.replace(
                'module.densenet121', 'densenet121'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.norm.1', '.norm1'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.norm.2', '.norm2'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.conv.1', '.conv1'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.conv.2', '.conv2'): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
            num_loaded = len(set(self.state_dict().keys())
                             & set(state_dict.keys()))
            num_total = len(state_dict.keys())
            if num_loaded < num_total:
                missing_params = set(state_dict.keys()).symmetric_difference(
                    set(self.state_dict().keys()))
                logger.info(
                    "Could not load these parameters due to name mismatch: " + str(missing_params))
            logger.info(
                f"Loaded {num_loaded}/{num_total} pretrained parameters")

        self.densenet121.classifier = nn.Identity()

    def forward(self, x1, x2=None):
        """
        """
        if x2 is None:
            return self.densenet121(x1)
        else:
            return self.densenet121(x1), self.densenet121(x2)


class ClusterModule(nn.Module):

    def __init__(self, in_features=1024, out_features=100):
        """
        """
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x1, x2=None):
        if x2 is None:
            return self.classifier(x1)
        else:
            return self.classifier(x1), self.classifier(x2)
