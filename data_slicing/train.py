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

import dataset as all_datasets
import transforms as custom_transforms
from transforms import transforms_ingredient
import modules
from util import ce_loss, output, get_sample_weights

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
    pretrain_imagenet = True
    pretrain_chexnet = False
    data_slicing = False

    assert not (pretrain_imagenet and pretrain_chexnet), \
        'pretrain_imagenet and pretrain_chexnet are mutually exclusive'

    assert cxr_only, \
        'CXR+segmentation input not yet implemented'

    assert not data_slicing, \
        'data slicing model not yet implemented'

    hypothesis_conditions = []

    # data slicing model or not
    if data_slicing:
        hypothesis_conditions.append('data_slicing')
    else:
        hypothesis_conditions.append('baseline')

    # feed in CXR as input or CXR+Segmentation
    if cxr_only:
        hypothesis_conditions.append('cxr_only')
    else:
        hypothesis_conditions.append('cxr_seg')

    # pretrain model with ImageNet, CheXNet, or none
    if pretrain_imagenet:
        hypothesis_conditions.append('pretrain_imagenet')
    elif pretrain_chexnet:
        hypothesis_conditions.append('pretrain_chexnet')
    else:
        hypothesis_conditions.append('no_pretrain')

    exp_dir = path.join('experiments', *hypothesis_conditions)

    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    task_to_label_dict = {t: t for t in tasks}
    task_to_cardinality_dict = {t: 2 for t in tasks}

    meta_config = {
        'device': 0
    }

    logging_config = {
        'evaluation_freq': 4000,
        'checkpointing': False,
    }

    path_to_images = '/lfs/1/jdunnmon/data/nih/images/images'
    path_to_labels = '/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/nih_labels.csv'
    dataset_configs = {
        'train': {
            'class_name': 'CheXNetDataset',
            'args': {
                'path_to_images': path_to_images,
                'path_to_labels': path_to_labels,
                'sample': 0,
                'seed': 1701,
                'finding': 'any',
                'transforms': transforms['augmentation'] + transforms['preprocessing'],
                'add_binary_triage_label': False
            }
        },
        'val': {
            'class_name': 'CheXNetDataset',
            'args': {
                'path_to_images': path_to_images,
                'path_to_labels': path_to_labels,
                'sample': 0,
                'seed': 1701,
                'finding': 'any',
                'transforms': transforms['preprocessing'],
                'add_binary_triage_label': False
            }
        }
    }

    dataloader_configs = {
        'train': {
            'batch_size': 16,
            'num_workers': 8,
            'shuffle': False
        },
        'val': {
            'batch_size': 16,
            'num_workers': 8,
            'shuffle': True
        }
    }
    sampler_configs = {}

    encoder_class = 'ClippedDenseNet'
    encoder_args = {
        'pretrained': True if pretrain_imagenet else False,
        'weights_path': 'model.pth.tar' if pretrain_chexnet else False
    }

    decoder_class = "LinearDecoder"
    decoder_args = {
        'num_layers': 1,
        'encoding_size': 1024,
        'dropout_p': 0.0
    }

    learner_config = {
        'n_epochs': 20,
        'valid_split': 'val',
        'optimizer_config': {'optimizer': 'sgd', 'lr': 0.001, 'l2': 0.000},
        'lr_scheduler_config': {
            'warmup_steps': None,
            'warmup_unit': 'batch',
            "lr_scheduler": "linear",
            "min_lr": 1e-6,
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
    def _init_meta(self, _run, _log, _seed, exp_dir, meta_config, learner_config, logging_config):
        is_unobserved = _run.meta_info['options']['--unobserve'] 
        
        # only if 'checkpointing' is defined, True, and the experiment is observed
        logging_config = dict(logging_config)
        logging_config['checkpointing'] = logging_config.get('checkpointing', False) and not is_unobserved
        
        emmental.init(path.join(exp_dir, '_emmental_logs'))
        Meta.update_config(
            config={
                'meta_config': {
                    **meta_config, 
                    'seed': _seed,
                },
                'model_config': {
                    'device': meta_config['device']
                },
                'learner_config': learner_config,
                'logging_config': logging_config
            }
        )
        ex.info.update({'emmental_log_path': Meta.log_path})
        _log.info(f'emmental_log_path set to {Meta.log_path}')

    @ex.capture
    def _init_datasets(self, _log, dataset_configs):
        datasets = {}
        for split in dataset_configs.keys():
            class_name = dataset_configs[split]['class_name']
            args = dataset_configs[split]['args']
            datasets[split] = getattr(all_datasets, class_name)(
                name=class_name,
                split=split,
                **args
            )
            _log.info(f'Loaded {split} split.')
        return datasets

    @ex.capture
    def _init_dataloaders(self, _log, dataloader_configs, task_to_label_dict):
        dataloaders = []
        for split in dataloader_configs.keys():
            dataloader_config = dataloader_configs[split]
            dataloader_config = {
                'sampler': self._init_sampler(split),
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
        sampler = None
        if split in sampler_configs:
            sampler_class = sampler_configs[split]['class_name']
            sampler_args = sampler_configs[split]['args']
            if sampler_class == 'WeightedRandomSampler':
                weights = get_sample_weights(
                    self.datasets[split], sampler_args['weight_task'], sampler_args['class_probs'])
                sampler = getattr(torch_data, sampler_class)(
                    weights=weights, num_samples=sampler_args['num_samples'], replacement=sampler_args['replacement'])
            elif sampler_class == 'RandomSampler':
                sampler = getattr(torch_data, sampler_class)(
                    data_source=self.datasets[split], **sampler_args)
            elif sampler_class != None:
                raise ValueError(
                    f'sampler_class {sampler_class} not recognized')

        if sampler:
            _log.info(f'Built sampler {sampler_class}.')
        else:
            _log.info(f'No sampler configured for {split} split.')
        return sampler

    @ex.capture
    def _init_model(self, _log, encoder_class, encoder_args,
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
                    metrics=['accuracy', 'roc_auc']),
            )
            for task_name in task_to_label_dict.keys()
        ]
        model = EmmentalModel(name='CheXNet', tasks=tasks)
        print(model)
        _log.info(f'Model initalized.')
        return model

    @ex.capture
    def run(self, _log):
        learner = EmmentalLearner()
        _log.info(f'Starting training.')
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
