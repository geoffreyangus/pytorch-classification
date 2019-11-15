from __future__ import print_function

import time
import os
import os.path as osp
from uuid import uuid4
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optimizers
import torch.optim.lr_scheduler as schedulers
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

from dataset.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from util import html
from models.scale_rcnn import SRCNN


EXPERIMENT_NAME = 'catheter-detection'
ex = Experiment(EXPERIMENT_NAME)

class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)

@ex.config
def config():
    """
    Config for catheter detector.
    """
    hypothesis_conditions = ['catheter_detector', 'synthetic_xray']
    exp_dir = osp.join('../experiments', *hypothesis_conditions)

    isTrain = False
    use_annot = True

    # path to images (should have subfolders trainA, trainB, valA, valB, etc)
    dataroot = '/Users/geoffreyangus/data/synthetic_xray'
    dataroot = '/lfs/1/gangus/data/synthetic_xray'
    # input batch size
    batchSize = 1
    # scale images to this size
    loadSize = 512
    # then crop to this size
    fineSize = 512
    # of input image channels
    input_nc = 3
    # of output image channels
    output_nc = 3
    # of gen filters in first conv layer
    ngf = 64
    # selects model to use for netG
    which_model_netG = 'srcnn'
    # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
    gpu_ids = [0]
    # name of the experiment. It decides where to store samples and models
    name = 'experiment_name'
    # chooses how datasets are loaded.
    dataset_mode = 'alignedsrcnn'
    # chooses which model to use
    model = 'srcnn'
    # AtoB or BtoA
    which_direction = 'AtoB'
    # threads for loading data
    nThreads = 2
    # models are saved here
    checkpoints_dir = './checkpoints'
    # instance normalization or batch normalization
    norm = 'instance'
    # if true, takes images in order to make batches, otherwise takes them randomly
    serial_batches = False
    # display window size
    display_winsize = 256
    # window id of the web display
    display_id = 1
    # visdom port of the web display
    display_port = 8097
    # no dropout for the generator
    no_dropout = False
    # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
    max_dataset_size = float("inf")
    # scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
    resize_or_crop = 'none'
    # if specified, do not flip the images for data augmentation
    no_flip = False
    # network initialization [normal|xavier|kaiming|orthogonal]
    init_type = 'normal'

    # testing only
    # # of test examples.
    ntest = float("inf")
    # saves results here.
    results_dir = './results/'
    # aspect ratio of result images
    aspect_ratio = 1.0
    # train, val, test, etc
    split = phase = 'test'
    # which epoch to load? set to latest to use latest cached model
    which_epoch = 'latest'
    # how many test images to run
    how_many = 100
    # internal or external
    sourceoftest = 'internal'

    # training only
    display_freq = 100
    display_single_pane_ncols = 0
    update_html_freq = 1000
    print_freq = 100
    save_latest_freq = 2000
    save_epoch_freq = 2
    continue_train = False
    epoch_count = 1
    phase = 'train'
    which_epoch = 'latest'
    niter = 200
    niter_decay = 100
    beta1 = 0.9
    lr = 0.0001
    no_html = True
    lr_policy = 'step'
    lr_decay_iters = 10

    # more testing boilerplate config
    nThreads = 1
    batchSize = 1
    serial_batches = True
    no_flip = True

class Harness:

    @ex.capture
    def __init__(self, _config):
        """
        """
        self.opts = Options(**_config)

        self.dataloader = self._init_dataloader()
        self.model = self._init_model()

        (self.visualizer,
         self.web_dir,
         self.webpage) = self._init_visualizer()

    @ex.capture
    def _init_dataloader(self, _config):
        dataloader = CreateDataLoader(self.opts)
        return dataloader

    @ex.capture
    def _init_model(self):
        model = SRCNN()
        model.initialize(self.opts)
        return model

    @ex.capture
    def _init_visualizer(self):
        visualizer = Visualizer(self.opts)
        # create website
        web_dir = os.path.join(self.opts.results_dir, self.opts.name, '%s_%s' % (self.opts.phase, self.opts.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opts.name, self.opts.phase, self.opts.which_epoch))
        return visualizer, web_dir, webpage

    @ex.capture
    def run(self, how_many, sourceoftest):
        dataset = self.dataloader.load_data()
        t = tqdm(total=len(dataset))
        for i, data in enumerate(dataset):
            if i >= how_many:
                break
            self.model.set_input(data)
            self.model.test()

            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))
            if sourceoftest == 'internal':
                self.visualizer.save_images(self.webpage, visuals, img_path)
            elif sourceoftest == 'external':
                self.visualizer.save_images_nogt(self.webpage, visuals, img_path)
            t.update()
        t.close()
        self.webpage.save()


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    ex.observers.append(FileStorageObserver(config['exp_dir']))


@ex.main
def main():
    harness = Harness()
    return harness.run()


if __name__ == "__main__":
    ex.run_commandline()
