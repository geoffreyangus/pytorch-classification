"""
Training script for CIFAR10/CIFAR100.

Modified for use with sacred.
"""
from __future__ import print_function

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

from dataset import CIFAR100, collate_train
import models.cifar as models
import losses.losses as losses
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

from sacred import Experiment
from sacred.observers import FileStorageObserver

EXPERIMENT_NAME = 'training'
ex = Experiment(EXPERIMENT_NAME)


@ex.config
def config():
    """
    Config for training harness.
    """
    cifar_type = 'CIFAR100'
    assert cifar_type in {'CIFAR10', 'CIFAR100'}, f'cifar_type {cifar_type}'

    hypothesis_conditions = [cifar_type, 'superclass']
    exp_dir = osp.join('experiments', *hypothesis_conditions)

    # meta
    data_dir = '/Users/geoffreyangus/data'       # on DAWN: '/lfs/1/gangus/data'
    data_dir = '/lfs/1/gangus/data'

    cuda = torch.cuda.is_available()
    device = 0 if cuda else 'cpu'

    checkpoint_dir = osp.join('checkpoints', str(uuid4()))
    num_epochs = 300

    # reload model
    resume = None                               # reload model path
    if resume:
        assert osp.isfile(resume), f'{resume} must be a valid model checkpoint'
        checkpoint_dir = osp.dirname(resume)
    evaluate = False

    # for subset analysis
    superclass = True
    superclass_config = {
        # string repr of subclass:subsample frac k:v pairs'
        'subsample_subclass': {},
        # string repr of subclass:whiten frac k:v pairs
        'whiten_subclass': {},
        # string repr of subclass_1:subclass_2 frac k:v pairs
        'diff_subclass': {}
    }

    # dataset args per split
    dataset_configs = {
        'train': {
            'transform': transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            'superclass': superclass,
            'subsample_subclass': superclass_config['subsample_subclass'],
            'whiten_subclass': superclass_config['whiten_subclass'],
            'diff_subclass': superclass_config['diff_subclass'],
        },
        'test': {
            'transform': transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            'superclass': superclass,
            'subsample_subclass': superclass_config['subsample_subclass'],
            'whiten_subclass': superclass_config['whiten_subclass'],
            'diff_subclass': superclass_config['diff_subclass']
        }
    }

    # dataloader args per split
    dataloader_configs = {
        'train': {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 4
        },
        'test': {
            'batch_size': 100,
            'shuffle': False,
            'num_workers': 4
        }
    }

    # model architecture
    if cifar_type == 'CIFAR10':
        num_classes = 10
    elif cifar_type == 'CIFAR100':
        num_classes = 20 if superclass else 100
    model_name = 'densenet'
    model_args = {
        'num_classes': num_classes,
        'depth': 100,
        'growthRate': 12,
        'compressionRate': 2,
        'dropRate': 0.0,
        'embeddings': True
    }

    # criterion config
    criterion_class = 'LMCL_loss'
    criterion_args = {
        'num_classes': num_classes,
        's': 7.00,
        'm': 0.2
    }

    # optimizer config
    optimizer_class = 'SGD'
    optimizer_args = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4                    # typically 1e-4
    }

    # scheduler config
    scheduler_class = None
    scheduler_args = {
        # epoch numbers to decrease learning rate
        'schedule': [150, 225],
        'gamma': 0.1                            # learning rate multiplied by gamma on schedule
    }


class TrainingHarness(object):

    def __init__(self):
        """
        Training harness for CIFAR dataset.
        """
        # optional state dict for manual controls
        self.state = {}
        self.best_acc = 0.0

        self._init_meta()
        self.datasets = self._init_datasets()
        self.dataloaders = self._init_dataloaders()
        self.model = self._init_model()
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    @ex.capture
    def _init_meta(self):
        cudnn.benchmark = True

    @ex.capture
    def _init_datasets(self, _log, cifar_type, superclass_config,
                       data_dir, dataset_configs):
        datasets = {}
        for split in ['train', 'test']:
            if cifar_type == 'CIFAR10':
                dataset_class = torchvision.datasets.CIFAR10
            else:
                dataset_class = CIFAR100
            datasets[split] = dataset_class(root=data_dir, train=(split == 'train'),
                                            download=False, **dataset_configs[split])
            _log.info(f'loaded {split} dataset for {cifar_type}')
            _log.info(f'length of {split} dataset: {len(datasets[split])}')
        return datasets

    @ex.capture
    def _init_dataloaders(self, _log, dataloader_configs):
        """
        TODO: use collate_test in order to measure accuracy on fine classes
        """
        dataloaders = {}
        for split in ['train', 'test']:
            dataloaders[split] = DataLoader(self.datasets[split],
                                            collate_fn=collate_train,
                                            **dataloader_configs[split])
            _log.info(f'loaded {split} dataloader')
        return dataloaders

    @ex.capture
    def _init_model(self, _log, model_name, model_args, device):
        model = models.__dict__[model_name](**model_args)
        model = torch.nn.DataParallel(model)
        if device != 'cpu':
            model = model.cuda(device)
        _log.info('total params: %.2fM' % (sum(p.numel()
                                               for p in model.parameters())/1000000.0))
        return model

    @ex.capture
    def _init_criterion(self, criterion_class, criterion_args, device):
        assert criterion_class == 'LMCL_loss', \
            'this file is for LMCL loss only due to training schematic'
        criterion_dict = {}
        criterion_dict['nll_loss'] = losses.CrossEntropyLoss()
        criterion_dict['lmcl_loss'] = getattr(losses, criterion_class)(feat_dim=self.model.module.inplanes, **criterion_args)
        if device != 'cpu':
            criterion_dict['lmcl_loss'] = criterion_dict['lmcl_loss'].cuda(device)
        return criterion_dict

    @ex.capture
    def _init_optimizer(self, optimizer_class, optimizer_args):
        optimizer_dict = {}
        optimizer_dict['nn'] = optimizers.SGD(self.model.parameters(), **optimizer_args)
        optimizer_dict['centers'] = optimizers.SGD(self.criterion['lmcl_loss'].parameters(), lr=0.01)
        return optimizer_dict

    @ex.capture
    def _init_scheduler(self, scheduler_class, scheduler_args):
        scheduler_dict = {}
        scheduler_dict['nn'] = schedulers.StepLR(self.optimizer['nn'], 150, gamma=0.1)
        scheduler_dict['centers'] = schedulers.StepLR(self.optimizer['centers'], 20, gamma=0.5)
        return scheduler_dict

    @ex.capture
    def run(self, _log, device, resume, checkpoint_dir, evaluate, num_epochs):
        if not os.path.isdir(checkpoint_dir):
            mkdir_p(checkpoint_dir)

        start_epoch = 0
        if resume:
            checkpoint = torch.load(resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(osp.join(checkpoint_dir, 'log.txt'), resume=True)
        else:
            logger = Logger(osp.join(checkpoint_dir, 'log.txt'))
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',
                              'Train Acc.', 'Valid Acc.'])

        if evaluate:
            _log.info('evaluation only')
            test_loss, test_acc = self.test(start_epoch, device)
            _log.info('test loss:  %.8f, test acc:  %.2f' %
                      (test_loss, test_acc))
            return {
                'test': {
                    'loss': test_loss,
                    'acc': test_acc
                }
            }

        # begin training
        for epoch in range(start_epoch, num_epochs):
            learning_rate = self.scheduler['nn'].get_lr()[0]
            _log.info('\nEpoch: [%d | %d] LR: %f' %
                      (epoch + 1, num_epochs, learning_rate))

            train_loss, train_acc = self.train(epoch, device)
            test_loss, test_acc = self.test(epoch, device)

            # append logger file
            logger.append([train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)

            self._save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'acc': test_acc,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

            self.scheduler['nn'].step()
            self.scheduler['centers'].step()

        logger.close()
        logger.plot()
        savefig(osp.join(self.checkpoint_dir, 'log.eps'))

        return {
            'train': {
                'loss': train_loss,
                'acc': train_acc
            },
            'test': {
                'loss': test_loss,
                'acc': test_acc
            }
        }

    @ex.capture
    def train(self, epoch, device):
        # switch to train mode
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t = tqdm(total=len(self.dataloaders['train']))
        for batch_idx, (inputs, targets) in enumerate(self.dataloaders['train']):
            if device != 'cpu':
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            inputs, targets = torch.autograd.Variable(
                inputs), torch.autograd.Variable(targets)

            # compute output
            emb, out = self.model(inputs)
            outputs, mlogits = self.criterion['lmcl_loss'](emb, targets)
            loss = self.criterion['nll_loss'](mlogits, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            if float(torch.__version__[:3]) < 0.5:
                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))
            else:
                losses.update(loss.data, inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer['nn'].zero_grad()
            self.optimizer['centers'].zero_grad()

            loss.backward()
            self.optimizer['nn'].step()
            self.optimizer['centers'].step()

            # plot progress
            t.set_postfix(
                loss='{:.3f}'.format(losses.avg.cpu().numpy()),
                top1='{:.3f}'.format(top1.avg.cpu().numpy()),
                top5='{:.3f}'.format(top5.avg.cpu().numpy()),
            )
            t.update()
        t.close()
        return (losses.avg, top1.avg)

    @ex.capture
    def test(self, epoch, device):
        # switch to evaluate mode
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t = tqdm(total=len(self.dataloaders['test']))
        for batch_idx, (inputs, targets) in enumerate(self.dataloaders['test']):
            if device != 'cpu':
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            inputs, targets = torch.autograd.Variable(
                inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            emb, out = self.model(inputs)
            outputs, mlogits = self.criterion['lmcl_loss'](emb, targets)
            loss = self.criterion['nll_loss'](mlogits, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            if float(torch.__version__[:3]) < 0.5:
                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))
            else:
                losses.update(loss.data, inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

            # plot progress
            t.set_postfix(
                loss='{:.3f}'.format(losses.avg.cpu().numpy()),
                top1='{:.3f}'.format(top1.avg.cpu().numpy()),
                top5='{:.3f}'.format(top5.avg.cpu().numpy()),
            )
            t.update()
        t.close()
        return (losses.avg, top1.avg)

    @ex.capture
    def _save_checkpoint(self, state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
        filepath = osp.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, osp.join(
                checkpoint_dir, 'model_best.pth.tar'))

        link_dir = osp.join(exp_dir, 'checkpoint')
        os.link(checkpoint_dir, link_dir)
        ex.add_artifact(link_dir)


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    ex.observers.append(FileStorageObserver(config['exp_dir']))


@ex.main
def main():
    harness = TrainingHarness()
    return harness.run()


if __name__ == "__main__":
    ex.run_commandline()
