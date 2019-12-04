import os
import os.path as osp

from PIL import Image
import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import transforms as custom_transforms
from util import compose

class MIMICDrainDetectionDataset(EmmentalDataset):
    
    def __init__(self, df_path, images_dir, split=None, transforms=None, cxr_only=False):
        assert cxr_only, 'CXR + seg not yet implemented'

        self.df = pd.read_csv(df_path, index_col=0)
        
        # use MIMIC split
        if split:
            self.df = self.df.loc[self.df['split'] == split]
            
        self.images_dir = images_dir
        self.cxr_only = cxr_only
        
        if transforms:
            self.transforms = {
                'x1': compose(transforms['x1']),
                'x2': compose(transforms['x2']),
                'joint': compose(transforms['joint'])
            }
        else:
            self.transforms = None

        X_dict = {'image_paths': []}
        Y_dict = {'drain': []}

        X_dict['image_paths'] = list(self.df['path_image'])
        Y_dict['drain'] = torch.from_numpy(np.array(list(self.df['drain']))).type(torch.LongTensor)

        EmmentalDataset.__init__(self, 'mimic-drain-detection-dataset', X_dict=X_dict, Y_dict=Y_dict, uid='image_paths')

    def __getitem__(self, idx):
        """
        """
        x_dict = {k: v[idx] for k, v in self.X_dict.items()}
        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}
    
        image_path = self.X_dict['image_paths'][idx]
        x1 = Image.open(osp.join(self.images_dir, image_path))
        if self.cxr_only:
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x = x1
        else:
            x1 = transforms.Grayscale(num_output_channels=1)(x1)
            x1 = custom_transforms.Unsqueeze(axis=-1)(x1)
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x2 = Image.open(osp.join(self.images_dir, f'{image_id[:-4]}_fake_B2.png'))
            x2 = custom_transforms.ChooseChannels(channels=[1, 2])(x2)
            if 'x2' in self.transforms:
                x2 = self.transforms['x2'](x2)
            x = torch.cat((x1, x2), axis=0)
            
        if 'joint' in self.transforms:
            x = self.transforms['joint'](x)

        x_dict['image'] = x
        return x_dict, y_dict
    

class DrainDetectionDataset(EmmentalDataset):

    def __init__(self, df_path, images_dir, split=None, transforms=None, cxr_only=False):
        """
        """
#         old arguments: (self, split_dir, split_str, images_dir, transforms=None, cxr_only=False)
#         split_path = osp.join(split_dir, 'split', f'{split_str}.csv')
#         self.split_df = pd.read_csv(split_path, index_col=0)
        self.split_df = pd.read_csv(df_path, index_col=0)
        self.images_dir = images_dir
        self.cxr_only = cxr_only

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
        if self.cxr_only:
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x = x1
        else:
            x1 = transforms.Grayscale(num_output_channels=1)(x1)
            x1 = custom_transforms.Unsqueeze(axis=-1)(x1)
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x2 = Image.open(osp.join(self.images_dir, f'{image_id[:-4]}_fake_B2.png'))
            x2 = custom_transforms.ChooseChannels(channels=[1, 2])(x2)
            if 'x2' in self.transforms:
                x2 = self.transforms['x2'](x2)
            x = torch.cat((x1, x2), axis=0)

        if 'joint' in self.transforms:
            x = self.transforms['joint'](x)

        x_dict = {k: v[idx] for k, v in self.X_dict.items() if k != 'image'}
        x_dict['image'] = x
        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}
        return x_dict, y_dict

