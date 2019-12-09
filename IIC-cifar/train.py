import sys
import os
import os.path as osp
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as torch_data

import emmental
from emmental import Meta
from emmental.data import EmmentalDataset
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

import datasets as all_datasets
import modules
from transforms import transforms_ingredient
from utils import ce_loss, output, pair_iic_loss, pair_output, compose, write_to_file

EXPERIMENT_NAME = 'trainer'
ex = Experiment(EXPERIMENT_NAME, ingredients=[transforms_ingredient])
ex.logger = logging.getLogger(__name__)
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

@ex.config
def config(transforms):
    """
    Config to train an IIC network on the CIFAR100 dataset.
    """
    # primary experiments

    clustering = False
    pair_type = None

    hypothesis_conditions = ['cifar_superclass']
    if clustering:
        assert pair_type is not None, \
            'if clustering, must specify a pairing_type'
        hypothesis_conditions += ['clustering', pair_type]
    else:
        hypothesis_conditions += ['baseline']

    exp_dir = osp.join('experiments', *hypothesis_conditions)

    # Emmental configs

    meta_config = {
        'verbose': True
    }

    model_config = {
        'model_path': None,
        'device': 0,
        'dataparallel': True
    }

    learner_config = {
        'n_epochs': 1,
        'valid_split': 'valid',
        'optimizer_config': {'optimizer': 'adam', 'lr': 0.01, 'l2': 0.000},
        'lr_scheduler_config': {}
    }

    logging_config = {
        'evaluation_freq': 1,
        'checkpointing': True,
        'checkpointer_config': {
            'checkpoint_metric': {
                'model/all/val/loss:min': 'max'
            }
        }
    }

    # data configs

    dataset_configs = {
        'train': {
            'class_name': 'IIC_CIFAR100',
            'args': {
                'root': '/lfs/1/gangus/data',
                'train': True,
                'transform': transforms['augmentation'] + transforms['preprocessing'],
                'target_transform': None,
                'download': False,
                'pair_type': pair_type,
                'pair_transform': transforms['g']
            }
        },
        'valid': {
            'class_name': 'IIC_CIFAR100',
            'args': {
                'root': '/lfs/1/gangus/data',
                'train': False,
                'transform': transforms['preprocessing'],
                'target_transform': None,
                'download': False,
                'pair_type': pair_type,
                'pair_transform': transforms['g']
            }
        }
    }

    dataloader_configs = {
        'train': {
            'batch_size': 16,
            'num_workers': 16,
            'shuffle': False
        },
        'valid': {
            'batch_size': 16,
            'num_workers': 16,
            'shuffle': False
        }
    }

    sampler_configs = {
        'train': {
            'class_name': 'RandomSampler',
            'args': {
                'num_samples': 64,
                'replacement': True
            }
        },
        'valid': {
            'class_name': 'RandomSampler',
            'args': {
                'num_samples': 64,
                'replacement': True
            }
        }
    }

    # architecture configs

    task_to_label_dict = {
        'superclass': 'superclass',
    }
    if clustering:
        task_to_label_dict['iic'] = 'superclass' # dummy label

    encoder_class = 'ClippedDenseNet'
    encoder_args = {
        'pretrained': False
    }

    decoder_class = "Linear"
    decoder_args = {
        'in_features': 1024,
        'out_features': 20
    }

    cluster_class = 'ClusterModule'
    cluster_args = {
        'in_features': 1024,
        'out_features': 100
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
    def _init_meta(self, _run, _log, _seed, exp_dir, meta_config, model_config,
                   learner_config, logging_config):
        is_unobserved = _run.meta_info['options']['--unobserved']

        # only if 'checkpointing' is defined, True, and the experiment is observed
        logging_config = dict(logging_config)
        logging_config['checkpointing'] = logging_config.get('checkpointing', False) \
                                          and not is_unobserved

        emmental.init(osp.join(exp_dir, '_emmental_logs'))
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
        for split in ['train', 'valid']:
            class_name = dataset_configs[split]['class_name']
            args = dataset_configs[split]['args']
            args = {
                **args,
                'transform': compose(args['transform']),
                'target_transform': compose(args['target_transform']),
                'pair_transform': compose(args['pair_transform'])
            }
            datasets[split] = getattr(all_datasets, class_name)(**args)
            _log.info(f'Loaded {split} split.')
        return datasets

    @ex.capture
    def _init_dataloaders(self, _log, dataloader_configs, task_to_label_dict):
        dataloaders = []
        for split in ['train', 'valid']:
            dataloader_config = dataloader_configs[split]
            dataloader_config = {
                **dataloader_config,
                'sampler': self._init_sampler(split)
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
        if split not in sampler_configs.keys():
            return None

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
    def _init_model(self,
                    encoder_class, encoder_args,
                    decoder_class, decoder_args,
                    cluster_class, cluster_args,
                    task_to_label_dict, clustering):
        encoder_module = getattr(modules, encoder_class)(**encoder_args)
        decoder_module = getattr(modules, decoder_class)(**decoder_args)
        if clustering:
            cluster_module = getattr(modules, cluster_class)(**cluster_args)
            tasks = [
                EmmentalTask(
                    name='superclass',
                    module_pool=nn.ModuleDict({
                        'encoder': encoder_module,
                        'decoder': decoder_module
                    }),
                    task_flow=[
                        {
                            'name': 'encoder',
                            'module': 'encoder',
                            'inputs': [('_input_', 'image_a'), ('_input_', 'image_b')]
                        },
                        {
                            'name': 'superclass_pred_head',
                            'module': 'decoder',
                            'inputs': [('encoder', 0)]
                        }
                    ],
                    loss_func=partial(ce_loss, 'superclass'),
                    output_func=partial(output, 'superclass'),
                    scorer=Scorer(metrics=["accuracy"]),
                ),
                EmmentalTask(
                    name='iic',
                    module_pool=nn.ModuleDict({
                        'encoder': encoder_module,
                        'cluster': cluster_module
                    }),
                    task_flow=[
                        {
                            'name': 'encoder',
                            'module': 'encoder',
                            'inputs': [('_input_', 'image_a'), ('_input_', 'image_b')]
                        },
                        {
                            'name': 'iic_pred_head',
                            'module': 'cluster',
                            'inputs': [('encoder', 0), ('encoder', 1)]
                        }
                    ],
                    loss_func=partial(pair_iic_loss, 'iic'),
                    output_func=partial(pair_output, 'iic'),
                    scorer=Scorer(metrics=[]),
                )
            ]
        else:
            tasks = [
                EmmentalTask(
                    name='superclass',
                    module_pool=nn.ModuleDict({
                        'encoder': encoder_module,
                        'decoder': decoder_module
                    }),
                    task_flow=[
                        {
                            'name': 'encoder',
                            'module': 'encoder',
                            'inputs': [('_input_', 'image_a')]
                        },
                        {
                            'name': 'superclass_pred_head',
                            'module': 'decoder',
                            'inputs': [('encoder', 0)]
                        }
                    ],
                    loss_func=partial(ce_loss, 'superclass'),
                    output_func=partial(output, 'superclass'),
                    scorer=Scorer(metrics=["accuracy"]),
                )
            ]
        model = EmmentalModel(name='iic-model', tasks=tasks)
        return model

    @ex.capture
    def run(self, _log):
        learner = EmmentalLearner()
        learner.learn(self.model, self.dataloaders)

        scores = self.model.score(self.dataloaders)
        _log.info(f"Metrics: {scores}")
        write_to_file("metrics.txt", scores)

        if Meta.config['logging_config']['checkpointing']:
            # Save best metrics into file
            _log.info(
                f"Best metrics: "
                f"{learner.logging_manager.checkpointer.best_metric_dict}"
            )
            write_to_file(
                "best_metrics.txt",
                learner.logging_manager.checkpointer.best_metric_dict,
            )

        _log.info(f"Logs written to {Meta.log_path}")


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
