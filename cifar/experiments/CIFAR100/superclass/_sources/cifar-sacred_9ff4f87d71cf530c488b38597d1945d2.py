"""
Training script for CIFAR10/CIFAR100.

Modified for use with sacred.
"""
from __future__ import print_function

import os
import os.path as osp
from uuid import uuid4
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dataset import CIFAR100, collate_train
import models.cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

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
    # data_dir = '/Users/geoffreyangus/data'       # on DAWN: '/lfs/1/gangus/data'
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

    dataloader_configs = {
        'train': {
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 4
        },
        'test': {
            'batch_size': 4,
            'shuffle': False,
            'num_workers': 4
        }
    }

    # optimizer args
    optimizer_args = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4                    # typically 1e-4
    }

    # scheduler args
    scheduler_args = {
        'schedule': [150, 225],                 # epoch numbers to decrease learning rate
        'gamma': 0.1                            # learning rate multiplied by gamma on schedule
    }

    if cifar_type == 'CIFAR10':
        num_classes = 10
    elif cifar_type == 'CIFAR100':
        num_classes = 20 if superclass else 100

    # model architecture
    model_name = 'densenet'                     # model architecture
    model_args = {
        'num_classes': num_classes,
        'depth': 100,
        'growthRate': 12,
        'compressionRate': 2,
        'dropRate': 0.0
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
    def _init_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    @ex.capture
    def _init_optimizer(self, optimizer_args):
        self.state['lr'] = optimizer_args['lr']
        optimizer = optim.SGD(self.model.parameters(), **optimizer_args)
        return optimizer

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
            test_loss, test_acc = self.test(self.dataloaders['test'], self.model,
                                            self.criterion, start_epoch, cuda)
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
            _log.info('\nEpoch: [%d | %d] LR: %f' %
                      (epoch + 1, num_epochs, self.state['lr']))

            train_loss, train_acc = self.train(epoch, device)
            test_loss, test_acc = self.test(epoch, device)

            # append logger file
            logger.append([state['lr'], train_loss,
                           test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': self.best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_dir)

            self.adjust_learning_rate(epoch)

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

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(self.dataloaders['train']))
        for batch_idx, (inputs, targets) in enumerate(self.dataloaders['train']):
            # measure data loading time
            data_time.update(time.time() - end)

            if device != 'cpu':
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            inputs, targets = torch.autograd.Variable(
                inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.dataloaders['train']),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    @ex.capture
    def test(self, epoch, device):
        # switch to evaluate mode
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=len(self.dataloaders['test']))
        for batch_idx, (inputs, targets) in enumerate(self.dataloaders['test']):
            # measure data loading time
            data_time.update(time.time() - end)

            if device != 'cpu':
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            inputs, targets = torch.autograd.Variable(
                inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.dataloaders['test']),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    @ex.capture
    def _save_checkpoint(self, state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
        filepath = osp.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, osp.join(
                checkpoint_dir, 'model_best.pth.tar'))

    @ex.capture
    def _adjust_learning_rate(self, epoch, scheduler_args):
        if epoch in schedule_args['schedule']:
            self.state['lr'] *= scheduler_args['gamma']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']


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
