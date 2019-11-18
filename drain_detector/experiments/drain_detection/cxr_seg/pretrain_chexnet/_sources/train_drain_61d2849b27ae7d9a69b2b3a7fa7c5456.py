import os
import os.path as path
import logging
from functools import partial

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torchvision import transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from dataset import DrainDetectionDataset
import transforms as custom_transforms
from transforms import transforms_ingredient
import modules
from util import ce_loss, output

EXPERIMENT_NAME = 'trainer'
ex = Experiment(EXPERIMENT_NAME, ingredients=[transforms_ingredient])
ex.logger = logging.getLogger(__name__)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config(transforms):
    """
    Configuration for training harness.
    """
    cxr_only = True
    pretrain_imagenet = False
    pretrain_chexnet = False
    assert not (pretrain_imagenet and pretrain_chexnet), \
        'pretrain_imagenet and pretrain_chexnet are mutually exclusive'
    
    hypothesis_conditions = ['drain_detection']
    
    # first hypothesis class
    if cxr_only:
        hypothesis_conditions.append('cxr_only')
    else:
        hypothesis_conditions.append('cxr_seg')
    
    # second hypothesis class
    if pretrain_imagenet:
        hypothesis_conditions.append('pretrain_imagenet')
    elif pretrain_chexnet:
        hypothesis_conditions.append('pretrain_chexnet')
    else:
        hypothesis_conditions.append('no_pretrain')
        
    exp_dir = path.join('experiments', *hypothesis_conditions)
    meta_config = {
        'device': 0
    }

    logging_config = {
        'evaluation_freq': 1,
        'checkpointing': True,
        'checkpointer_config': {
            'checkpoint_metric': {
                'drain/drain-detection-dataset/valid/roc_auc': 'max'
            }
        }
    }

    images_dir = '/lfs/1/gangus/repositories/pytorch-classification/catheter_detector/results/catheter_detect/test_latest/images'
    split_dir = '/lfs/1/gangus/repositories/pytorch-classification/drain_detector/data/by-patient-id'
    dataset_configs = {
        'train': {
            'class_name': 'DrainDetectionDataset',
            'args': {
                'split_dir': split_dir,
                'images_dir': images_dir,
                'transforms': {
                    'x1': transforms['preprocessing']['x1'] + transforms['augmentation']['x1'],
                    'x2': transforms['preprocessing']['x2'] + transforms['augmentation']['x2'],
                    'joint': transforms['preprocessing']['joint'] + transforms['augmentation']['joint']
                },
                'cxr_only': cxr_only
            }
        },
        'valid': {
            'class_name': 'DrainDetectionDataset',
            'args': {
                'split_dir': split_dir,
                'images_dir': images_dir,
                'transforms': {
                    'x1': transforms['preprocessing']['x1'],
                    'x2': transforms['preprocessing']['x2'],
                    'joint': transforms['preprocessing']['joint']
                },
                'cxr_only': cxr_only
            }
        }
    }

    dataloader_configs = {
        'train': {
            'batch_size': 16,
            'num_workers': 8,
            'shuffle': False
        },
        'valid': {
            'batch_size': 20,
            'num_workers': 8,
            'shuffle': True
        }
    }

    sampler_configs = {
        'train': {
            'class_name': 'RandomSampler',
            'args': {
                'num_samples': 800,
                'replacement': True,
            }
        }
    }

    task_to_label_dict = {
        'drain': 'drain',
    }
    
    task_to_cardinality_dict = {
        'drain': 2
    }
    
    encoder_class = 'ClippedDenseNet'
    encoder_args = {
        'pretrained': True if pretrain_imagenet else False,
        'weights_path': 'model.pth.tar' if pretrain_chexnet else False
    }

    decoder_class = "LinearDecoder"
    decoder_args = {
        'num_layers': 2,
        'encoding_size': 1024,
        'dropout_p': 0.0
    }

    learner_config = {
        'n_epochs': 100,
        'valid_split': 'valid',
        'optimizer_config': {'optimizer': 'adam', 'lr': 0.01, 'l2': 0.000},
        'lr_scheduler_config': {
            'warmup_steps': None,
            'warmup_unit': 'batch',
            'lr_scheduler': 'step',
            'step_config': {
                'step_size': 6,
                'gamma': 0.5
            }
        },
    }


class TrainingHarness(object):

    def __init__(self):
        """
        """
        self._init_meta()

        self.datasets = self._init_datasets()
        self.dataloaders = self._init_dataloaders()
        self.model = self._init_model()

    @ex.capture
    def _init_meta(self, _seed, exp_dir, meta_config, learner_config, logging_config):
        emmental.init(path.join(exp_dir, '_emmental_logs'))
        Meta.update_config(
            config={
                'meta_config': {**meta_config, 'seed': _seed},
                'model_config': {'device': meta_config['device']},
                'learner_config': learner_config,
                'logging_config': logging_config
            }
        )

    @ex.capture
    def _init_datasets(self, _log, dataset_configs):
        datasets = {}
        for split in ['train', 'valid']:
            class_name = dataset_configs[split]['class_name']
            args = dataset_configs[split]['args']
            datasets[split] = DrainDetectionDataset(
                split_str=split,
                **args
            )
            _log.info(f'Loaded {split} split.')
        return datasets

    @ex.capture
    def _init_dataloaders(self, _log, dataloader_configs, task_to_label_dict):
        dataloaders = []
        for split in ['train', 'valid']:
            dataloader_config = dataloader_configs[split]
            if split == 'train':
                sampler = self._init_sampler(split)
                dataloader_config = {
                    'sampler': sampler,
                    **dataloader_config
                }
            dl = EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=self.datasets[split],
                split=split,
                **dataloader_config,
            )
            dataloaders.append(dl)
            _log.info(f'Built dataloader for {split} set.')
        return dataloaders

    @ex.capture
    def _init_sampler(self, split, _log, sampler_configs):
        sampler_class = sampler_configs[split]['class_name']
        sampler_args = sampler_configs[split]['args']
        if sampler_class == 'WeightedRandomSampler':
            weights = get_sample_weights(
                self.datasets[split], sampler_args['weight_task'], sampler_args['class_probs'])
            sampler = getattr(torch_data, sampler_class)(
                weights=weights, num_samples=sampler_args['num_samples'], replacement=sampler_args['replacement'])
        else:
            sampler = getattr(torch_data, sampler_class)(
                data_source=self.datasets[split], **sampler_args)
        _log.info(f'Built sampler {sampler_class}.')
        return sampler

    @ex.capture
    def _init_model(self, encoder_class, encoder_args,
                          decoder_class, decoder_args, 
                          task_to_label_dict, task_to_cardinality_dict):
        encoder_module = getattr(modules, encoder_class)(**encoder_args)
        tasks = [
            EmmentalTask(
                name=task_name,
                module_pool=nn.ModuleDict(
                    {
                        f'encoder_module': encoder_module,
                        f'decoder_module_{task_name}': getattr(modules, decoder_class)(task_to_cardinality_dict[task_name], **decoder_args),
                    }
                ),
                task_flow=[
                    {
                        'name': 'encoder_module', 'module': 'encoder_module', 'inputs': [('_input_', 'image')]
                    },
                    {
                        'name':   f'decoder_module_{task_name}',
                        'module': f'decoder_module_{task_name}',
                        'inputs': [('encoder_module', 0)],
                    },
                ],
                loss_func=partial(ce_loss, task_name),
                output_func=partial(output, task_name),
                scorer=Scorer(
                    metrics=['accuracy', 'roc_auc', 'precision', 'recall', 'f1']),
            )
            for task_name in task_to_label_dict.keys()
        ]
        model = EmmentalModel(name='drain-detection-model', tasks=tasks)
        return model

    def run(self):
        learner = EmmentalLearner()
        learner.learn(self.model, self.dataloaders)


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    ex.observers.append(FileStorageObserver(config['exp_dir']))


@ex.main
def main():
    trainer = TrainingHarness()
    res = trainer.run()
    return res


if __name__ == '__main__':
    ex.run_commandline()
