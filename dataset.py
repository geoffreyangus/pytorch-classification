from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class CIFAR100(VisionDataset):
    """`CIFAR109 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, superclass=True):

        super(CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.superclass = superclass # use superclass labels for training

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # Setting meta
        if self.superclass:
            self.target_key = 'coarse_label',
            print('Using coarse labels...')
        else:
            self.target_key = 'fine_label'
            print('Using fine labels...')

        self.meta = {
        'filename': 'meta',
        'target_key': f'{self.target_key}_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
            }   

        file_path = os.path.join(self.root, self.base_folder, 'train')
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                self.data = pickle.load(f)
            else:
                self.data = pickle.load(f, encoding='latin1')

        self.data['data'] = np.vstack(self.data['data']).reshape(-1, 3, 32, 32)
        self.data['data'] = self.data['data'].transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['target_key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        item_data = {
                'fine_label':self.data['fine_labels'][index],
                'coarse_label':self.data['coarse_labels'][index],
                'filename':self.data['filenames'][index],
                }

        img, target = self.data['data'][index], item_data[self.target_key]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        item_data['target'] = target
        item_data['img'] = img

        return item_data


    def __len__(self):
        return len(self.data['data'])

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def collate_train(batch):
    imgs = torch.stack([a['img'] for a in batch])
    targets = torch.LongTensor([a['target'] for a in batch])
    return imgs, targets
