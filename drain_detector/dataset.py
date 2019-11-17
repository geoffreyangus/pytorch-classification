import os
import os.path as osp

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class DrainDetectorDataset(Dataset)

    def __init__(self, split_dir, split_str, images_dir, transforms=None):
        """
        """
        split_path = osp.join(split_dir, 'split', f'{split_str}.csv')
        self.split_df = pd.read_csv(split_path, index_col=0)
        self.images_dir = images_dir

        self.image_ids = list(split_df['Image Index'])
        self.labels = list(split_df['drain'])

        if transforms:
            self.transforms = {
                'x1': compose(transforms['x1']),
                'x2': compose(transforms['x2']),
                'joint': compose(transforms['joint'])
            }
        else:
            self.transforms = None

    def __getitem__(self, idx):
        """
        """
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        image_id = self.ids[idx]
        x1 = Image.open(osp.join(images_dir, f'{image_id}_real_A2.png'))
        if 'x1' in self.transforms:
            x1 = self.transforms['x1'](x1)

        x2 = Image.open(osp.join(images_dir, f'{image_id}_fake_B2.png'))
        if 'x2' in self.transforms:
            x2 = self.transforms['x2'](x2)

        x = torch.cat((x1, x2), axis=-1)
        if 'x' in self.transforms:
            x = self.transforms['joint'](x)

        y = self.labels[idx]
        return x, y

