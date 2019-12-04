import os
import os.path as path
import logging
from functools import partial
from collections import defaultdict

import emmental
from emmental import Meta
from emmental.contrib import slicing
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

import pandas as pd
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
import slicing_functions
from util import ce_loss, output, get_sample_weights, write_to_file

EXPERIMENT_NAME = 'trainer'
ex = Experiment(EXPERIMENT_NAME, ingredients=[transforms_ingredient])
ex.logger = logging.getLogger(__name__)
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

CXR8_TASKS = [
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

@ex.config
def config(transforms):
    """
    Configuration for training harness.
    """
    cxr_only = True
    pretrain_imagenet = True
    pretrain_chexnet = False
    slice_enabled = False
    task_str = 'CXR8'
    num_samples = 0

    assert not (pretrain_imagenet and pretrain_chexnet), \
        'pretrain_imagenet and pretrain_chexnet are mutually exclusive'

    assert cxr_only, \
        'CXR+segmentation input not yet implemented'

    if task_str == 'CXR8':
        task_names = CXR8_TASKS
        add_binary_triage_label = False
    elif task_str == 'TRIAGE':
        task_names = ['Abnormal']
        add_binary_triage_label = True
    else:
        task_names = task_str.split('&')
        for task in task_names:
            assert task in CXR8_TASKS, f'task {task} not in CXR8_TASKS'

    slice_task_names = {t: {'drain': 'slice_drain'} for t in task_names if t == 'Pneumothorax'}
    slice_task_names_eval = slice_task_names
    if slice_task_names or slice_task_names_eval:
        assert slice_enabled, 'slice_tasks only if slice_enabled'
    slice_df_path = '/lfs/1/gangus/repositories/pytorch-classification/drain_detector/data/by-patient-id/split/all.csv'
    slice_dropout = 0.0

    assert not num_samples < 0, \
        'num_samples must be a non-negative number'

    hypothesis_conditions = []

    # data slicing model or not
    if slice_enabled:
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

    # use task_str as directory name
    hypothesis_conditions.append(task_str)

    # restrict number of samples
    if num_samples > 0:
        hypothesis_conditions.append(f'{num_samples}_samples')
    else:
        hypothesis_conditions.append(f'full_dataset')

    exp_dir = path.join('experiments', *hypothesis_conditions)

    task_to_label_dict = {t: t for t in task_names}
    task_to_cardinality_dict = {t: 2 for t in task_names}

    path_to_images = '/lfs/1/jdunnmon/data/nih/images/images'
    path_to_labels = '/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/nih_labels.csv'
    dataset_configs = {
        'train': {
            'class_name': 'CheXNetDataset',
            'args': {
                'path_to_images': path_to_images,
                'path_to_labels': path_to_labels,
                'sample': num_samples,
                'seed': 1701,
                'finding': 'any',
                'transforms': transforms['augmentation'] + transforms['preprocessing'],
                'add_binary_triage_label': add_binary_triage_label
            }
        },
        'val': {
            'class_name': 'CheXNetDataset',
            'args': {
                'path_to_images': path_to_images,
                'path_to_labels': path_to_labels,
                'sample': num_samples,
                'seed': 1701,
                'finding': 'any',
                'transforms': transforms['preprocessing'],
                'add_binary_triage_label': add_binary_triage_label
            }
        }
    }

    dataloader_configs = {
        'train': {
            'batch_size': 16,
            'num_workers': 16,
            'shuffle': True
        },
        'val': {
            'batch_size': 16,
            'num_workers': 16,
            'shuffle': False
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

    # Additional Emmental Meta configs

    meta_config = {
        'verbose': True,
        'log_path': None
    }

    data_config = {
        'min_data_len': 0,
        'max_data_len': 0
    }

    model_config = {
        'model_path': None,
        'device': 0,
        'dataparallel': True
    }

    logging_config = {
        'evaluation_freq': 1,
        'checkpointing': False,
        'checkpointer_config': {
            'checkpoint_metric': {
                "model/val/accuracy": "max"
            }
        }
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

        (self.slicing_functions,
         self.slicing_functions_eval) = self._init_slicing_functions()
        self.model = self._init_model()

    @ex.capture
    def _init_meta(self, _run, _log, _seed, exp_dir,
                   meta_config, model_config,
                   learner_config, logging_config):
        is_unobserved = _run.meta_info['options']['--unobserved']

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
                    **model_config
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
                task_to_label_dict={**task_to_label_dict},
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
    def _init_model(self, _log):
        tasks = self._init_tasks()
        model = EmmentalModel(name='CheXNet', tasks=tasks)
        if Meta.config["model_config"]["model_path"]:
            model.load(Meta.config["model_config"]["model_path"])
        _log.info(f'Model initalized.')
        return model

    @ex.capture
    def _init_tasks(self, _log, encoder_class, encoder_args,
                    decoder_class, decoder_args,
                    task_to_label_dict, task_to_cardinality_dict, slice_dropout):
        encoder_module = getattr(modules, encoder_class)(**encoder_args)
        tasks = [
            EmmentalTask(
                name=task_name,
                module_pool=nn.ModuleDict(
                    {
                        f'encoder_module': encoder_module,
                        f'decoder_module_{task_name}': nn.Linear(in_features=decoder_args['encoding_size'],
                                                                 out_features=task_to_cardinality_dict[task_name]),
                    }
                ),
                task_flow=[
                    {
                        'name': 'encoder_module',
                        'module': 'encoder_module',
                        'inputs': [('_input_', 'image')]
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
                    metrics=['accuracy', 'f1']),
            )
            for task_name in task_to_label_dict.keys()
        ]
        all_tasks = []
        for task in tasks:
            if task.name in self.slicing_functions.keys():
                slice_distribution = slicing.add_slice_labels(
                    task,
                    self.dataloaders,
                    self.slicing_functions[task.name]
                )
                slice_tasks = slicing.build_slice_tasks(
                    task,
                    self.slicing_functions[task.name],
                    slice_distribution,
                    dropout=slice_dropout,
                    slice_ind_head_module=None,
                )
                all_tasks.extend(slice_tasks)
            else:
                all_tasks.append(task)
        tasks = all_tasks
        return tasks

    @ex.capture
    def _init_slicing_functions(self, _log, slice_df_path, slice_task_names, slice_task_names_eval):
        slicing_functions.SLICE_DF = pd.read_csv(slice_df_path).set_index('Image Index')

        slicing_functions_model = defaultdict(dict)
        for slice_task_name, slice_func_dict in slice_task_names.items():
            slicing_functions_model[slice_task_name] = {k: getattr(slicing_functions, v)
                                                        for k, v in slice_func_dict.items()}
        slicing_functions_eval = defaultdict(dict)
        for slice_task_name, slice_func_dict in slice_task_names_eval.items():
            slicing_functions_eval[slice_task_name] = {k: getattr(slicing_functions, v)
                                                       for k, v in slice_func_dict.items()}

        return slicing_functions_model, slicing_functions_eval

    @ex.capture
    def run(self, _log, task_names):
        _log.info(f"Emmental config: {Meta.config}")
        write_to_file("emmental_config.txt", Meta.config)

        learner = EmmentalLearner()
        _log.info(f'Starting training.')
        learner.learn(self.model, self.dataloaders)
        _log.info(f'Finished training.')

        slice_func_dict = {}
        for t in task_names:
            slice_func_dict.update(self.slicing_functions_eval[t])

        if len(slice_func_dict) > 0:
            scores = score_slices(self.model, self.dataloaders, task_names, slice_func_dict)
        else:
            scores = self.model.score(self.dataloaders)

        # Save metrics into file
        _log.info(f'Metrics: {scores}')
        write_to_file('metrics.txt', scores)

        # Save best metrics into file
        _log.info(f'Best metrics: {learner.logging_manager.checkpointer.best_metric_dict}')
        write_to_file('best_metrics.txt', learner.logging_manager.checkpointer.best_metric_dict)


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
