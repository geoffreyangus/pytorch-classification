"""
Training script for CIFAR10/CIFAR100.

Modified for use with sacred.
"""
import os
import os.path as osp
from uuid import uuid4

from sacred import Experiment
from sacred.observers import FileStorageObserver

EXPERIMENT_NAME = 'training'
ex = Experiment(EXPERIMENT_NAME)

@ex.config
def config():
    """
    Config for training harness.
    """
    hypothesis_conditions = ['cifar100', 'superclass']
    exp_dir = osp.join('experiments', *hypothesis_conditions)

    # meta
    data_dir = 'Users/geoffreyangus/data'       # on DAWN: '/lfs/1/gangus/data'
    device = 'cpu'

    checkpoint = osp.join('checkpoints', str(uuid4()))
    num_epochs = 300

    # reload model
    resume = None                               # reload model path
    if resume:
        assert osp.isfile(resume), f'{resume} must be a valid model checkpoint'
        checkpoint = osp.dirname(resume)
    start_epoch = 0                             # the epoch number to start on (useful on restarts)
    evaluate = False

    # for subset analysis
    superclass = True
    superclass_config = {
        'subsample_subclass': {},               # string repr of subclass:subsample frac k:v pairs'
        'whiten_subclass': {},                  # string repr of subclass:whiten frac k:v pairs
        'diff_subclass': {}                     # string repr of subclass_1:subclass_2 frac k:v pairs
    }

    dataset_configs = {
        'train': {
            'superclass': superclass,
            'subsample_subclass': superclass_config['subsample_subclass'],
            'whiten_subclass': superclass_config['whiten_subclass'],
            'diff_subclass': superclass_config['diff_subclass']
        },
        'test': {
            'superclass': superclass,
            'subsample_subclass': superclass_config['subsample_subclass'],
            'whiten_subclass': superclass_config['whiten_subclass'],
            'diff_subclass': superclass_config['diff_subclass']
        }
    }

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

    # learning rate
    learning_rate = 0.1
    schedule = [150, 225]                       # epoch numbers to decrease learning rate
    gamma = 0.1                                 # learning rate multiplied by gamma on schedule
    momentum = 0.9

    # regularization
    weight_decay = 1e-4                         # weight decay (typically 1e-4)
    dropout = 0.0

    # model architecture
    model_name = 'densenet'                     # model architecture
    model_args = {
        'depth': 100,
        'growth_rate': 12,
        'compression_rate': 2
    }


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    ex.observers.append(FileStorageObserver(config['exp_dir']))



@ex.main
def main():
    print('hello')


if __name__ == "__main__":
    ex.run_commandline()
