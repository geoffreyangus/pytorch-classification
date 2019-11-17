import os
import os.path as osp

from PIL import Image
import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from torch.utils.data import Dataset
from util import compose


class DrainDetectionDataset(EmmentalDataset):

    def __init__(self, split_dir, split_str, images_dir, transforms=None):
        """
        """
        split_path = osp.join(split_dir, 'split', f'{split_str}.csv')
        self.split_df = pd.read_csv(split_path, index_col=0)
        self.images_dir = images_dir

        if transforms:
            self.transforms = {
                'x1': compose(transforms['x1']),
                'x2': compose(transforms['x2']),
                'joint': compose(transforms['joint'])
            }
        else:
            self.transforms = None
            
        X_dict = {'image_ids': []}
        Y_dict = {'drain': []}

        X_dict['image_ids'] = list(self.split_df['Image Index'])
        Y_dict['drain'] = torch.from_numpy(np.array(list(self.split_df['drain'])))
        
        EmmentalDataset.__init__(self, 'drain-detection-dataset', X_dict=X_dict, Y_dict=Y_dict)

    def __getitem__(self, idx):
        """
        """
        image_id = self.X_dict['image_ids'][idx]
        x1 = Image.open(osp.join(self.images_dir, f'{image_id[:-4]}_real_A2.png'))
        if 'x1' in self.transforms:
            x1 = self.transforms['x1'](x1)
        x2 = Image.open(osp.join(self.images_dir, f'{image_id[:-4]}_fake_B2.png'))
        if 'x2' in self.transforms:
            x2 = self.transforms['x2'](x2)
        
        x = torch.cat((x1, x2), axis=0)
        if 'x' in self.transforms:
            x = self.transforms['joint'](x)
        
        x_dict = {k: v[idx] for k, v in self.X_dict.items() if k != 'image'}
        x_dict['image'] = x
        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}        
        return x_dict, y_dict

