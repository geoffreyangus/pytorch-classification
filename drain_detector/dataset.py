import os
import os.path as osp
import logging

from PIL import Image
import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import transforms as custom_transforms
from util import compose

logger = logging.getLogger(__name__)


class MIMICDrainDetectionDataset(EmmentalDataset):

    def __init__(self, df_path, images_dir, split=None, transforms=None, cxr_only=False, frontal_only=True):
        """Dataset for drain detection using the MIMIC dataset.

        Args:
            df_path (string): path of CSV containing labels and image paths.
            images_dir (string): base directory of images.
            split (string, optional): optional split string to filter CSV.
            transforms (dict, optional): Dict[str, callable] mapping transforms
                to different components of the final input image. Dataset
                implemented to permit different transforms to be applied to
                x1 (original CXR), x2 (predicted segmentation), and/or joint (CXR + seg)
            cxr_only (bool, optional): If True, treats the CXR as the whole
                input image. Else, treats CXR as a single channel and concatenates
                it with a predicted segmentation.
            frontal_only (bool, optional): If True, filters the CSV for frontal
                images, posterioranterior (PA) or anteroposterior (AP) positions.
                CXR8 dataset uses frontal images only, so useful for training
                models with high transferability.
        """
        assert cxr_only, 'CXR + seg not yet implemented'

        self.df = pd.read_csv(df_path, index_col=0)

        # use orientation found in CheXNet (PA and AP)
        if frontal_only:
            frontal_df = self.df.loc[self.df['ViewPosition'].isin({
                                                                  'PA', 'AP'})]
            logger.info(
                f'frontal only. using {len(frontal_df)} of {len(self.df)} images')
            self.df = frontal_df

        # use MIMIC split
        if split:
            split_df = self.df.loc[self.df['split'] == split]
            logger.info(
                f'using {split} split ({len(split_df)} of {len(self.df)} images)')
            self.df = split_df

        self.images_dir = images_dir
        self.cxr_only = cxr_only

        if transforms:
            self.transforms = {
                'x1': transforms['x1'],
                # 'x2': compose(transforms['x2']),
                'joint': transforms['joint']
            }
        else:
            self.transforms = None

        X_dict = {'image_paths': []}
        Y_dict = {'drain': []}

        X_dict['image_paths'] = list(self.df['path_image'])
        Y_dict['drain'] = torch.from_numpy(
            np.array(list(self.df['drain']))).type(torch.LongTensor)

        EmmentalDataset.__init__(
            self, 'mimic-drain-detection-dataset', X_dict=X_dict, Y_dict=Y_dict, uid='image_paths')

    def __getitem__(self, idx):
        """Returns either the original or feature-augmented CXR image.

        If cxr_only==True, returns a 3-channel CXR image. Else, returns a
        concatenation of two image sources: (1) a 1-channel, greyscale
        representation of CXR image and (2) a 2-channel representation of a
        predicted segmentation, where the first channel represents predicted
        catheters and the second channel represents predicted annotations.

        Source for predicted segmentation:
        https://github.com/xinario/catheter_detection

        Args:
            index (int): Index

        Returns:
            tuple: (x: dict, y: dict) where x is a dictionary mapping all
                possible inputs to EmmentalModel and y is a dictionary for all
                possible labels to EmmentalTasks.
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
            x2 = Image.open(
                osp.join(self.images_dir, f'{image_id[:-4]}_fake_B2.png'))
            x2 = custom_transforms.ChooseChannels(channels=[1, 2])(x2)
            if 'x2' in self.transforms:
                x2 = self.transforms['x2'](x2)
            x = torch.cat((x1, x2), axis=0)

        if 'joint' in self.transforms:
            x = self.transforms['joint'](x)

        x_dict['image'] = x
        return x_dict, y_dict


class DrainDetectionDataset(EmmentalDataset):

    def __init__(self, df_path, images_dir, transforms=None, cxr_only=False):
        """Dataset for drain detection using the output of the segmentation model.

        Segmentation model used to generate images for use with this dataset:
        https://github.com/xinario/catheter_detection

        Args:
            df_path (string): path of CSV containing labels and image paths.
                NOTE: there is no split argument for these dataframes. Use
                different CSVs for each split.
            images_dir (string): base directory of images.
            transforms (dict, optional): Dict[str, callable] mapping transforms
                to different components of the final input image. Dataset
                implemented to permit different transforms to be applied to
                x1 (original CXR), x2 (predicted segmentation), and/or joint (CXR + seg)
            cxr_only (bool, optional): If True, treats the CXR as the whole
                input image. Else, treats CXR as a single channel and concatenates
                it with a predicted segmentation.
        """
#         old arguments: (self, split_dir, split_str, images_dir, transforms=None, cxr_only=False)
#         split_path = osp.join(split_dir, 'split', f'{split_str}.csv')
#         self.split_df = pd.read_csv(split_path, index_col=0)
        self.split_df = pd.read_csv(df_path, index_col=0)
        self.images_dir = images_dir
        self.cxr_only = cxr_only

        if transforms:
            self.transforms = {
                'x1': transforms['x1'],
                'x2': transforms['x2'],
                'joint': transforms['joint')
            }
        else:
            self.transforms = None

        X_dict = {'image_ids': []}
        Y_dict = {'drain': []}

        X_dict['image_ids'] = list(self.split_df['Image Index'])
        Y_dict['drain'] = torch.from_numpy(
            np.array(list(self.split_df['drain'])))

        EmmentalDataset.__init__(
            self, 'drain-detection-dataset', X_dict=X_dict, Y_dict=Y_dict)

    def __getitem__(self, idx):
        """Returns either the original or feature-augmented CXR image.

        If cxr_only==True, returns a 3-channel CXR image. Else, returns a
        concatenation of two image sources: (1) a 1-channel, greyscale
        representation of CXR image and (2) a 2-channel representation of a
        predicted segmentation, where the first channel represents predicted
        catheters and the second channel represents predicted annotations.

        Source for predicted segmentation:
        https://github.com/xinario/catheter_detection

        Args:
            index (int): Index

        Returns:
            tuple: (x: dict, y: dict) where x is a dictionary mapping all
                possible inputs to EmmentalModel and y is a dictionary for all
                possible labels to EmmentalTasks.
        """
        image_id = self.X_dict['image_ids'][idx]
        x1 = Image.open(
            osp.join(self.images_dir, f'{image_id[:-4]}_real_A2.png'))
        if self.cxr_only:
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x = x1
        else:
            x1 = transforms.Grayscale(num_output_channels=1)(x1)
            x1 = custom_transforms.Unsqueeze(axis=-1)(x1)
            if 'x1' in self.transforms:
                x1 = self.transforms['x1'](x1)
            x2 = Image.open(
                osp.join(self.images_dir, f'{image_id[:-4]}_fake_B2.png'))
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
