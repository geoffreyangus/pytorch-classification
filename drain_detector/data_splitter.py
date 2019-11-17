import os
import os.path as osp
from collections import Counter, defaultdict, deque
import random

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
import yaml

import util


EXPERIMENT_NAME = 'splitter'
ex = Experiment(EXPERIMENT_NAME)


@ex.config
def config():
    """
    """
    data_dir = '/lfs/1/gangus/data/chexnet/CXR8-ORIG-DRAIN-SLICE-DATA/drain_detection'

    hypothesis_conditions = ['by-patient-id']
    exp_dir = osp.join('data', 'split', *hypothesis_conditions)

    strata_key = 'Patient ID'
    item_key = 'Image Index'
    item_label = 'drain'

    split_to_count = {
        'train': 0,  # will be replaced in the next line
        'valid': 100,
        'test': 0
    }
    split_to_count['train'] = len(pd.read_csv(osp.join(
        data_dir, 'attrs.tsv'), sep='\t', index_col=0).index.unique()) - sum(split_to_count.values())
    train_split = 'train'


class DataSplitter:

    def __init__(self):
        """
        """
        self.metadata = {}

    @ex.capture
    def run(self, exp_dir, data_dir, hypothesis_conditions, split_to_count, strata_key, _log):
        """
        """
        self.metadata.update({
            'meta.data_dir': data_dir,
            'meta.split_to_count': dict(split_to_count),
            'meta.strata_key': strata_key
        })

        attrs_path = osp.join(data_dir, 'attrs.csv')
        attrs_df = pd.read_csv(attrs_path, index_col=0)

        split_to_quota = self._analyze(attrs_df)
        strata_id_to_items = self._shuffle(attrs_df)
        split_to_item_ids = self._assign(strata_id_to_items, split_to_quota)

        split_to_split_df = self._format(attrs_df, split_to_item_ids)
        for split, split_df in split_to_split_df.items():
            split_df_path = osp.join(exp_dir, f'{split}.csv')
            _log.info(f'saving to {split_df_path}')
            split_df.to_csv(split_df_path)
            ex.add_artifact(split_df_path)

        metadata_path = osp.join(exp_dir, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            f.write(yaml.dump(self.metadata))
        ex.add_artifact(metadata_path)

    @ex.capture
    def _analyze(self, attrs_df, split_to_count, item_key, item_label):
        """
        """
        labels = []
        item_groups = attrs_df.groupby(item_key)
        for item_id, item_group in item_groups:
            labels.append(
                list(set(item_group[item_label]))[0])
        label_freqs = sorted(Counter(labels).items())

        split_to_quota = defaultdict(dict)
        for label, freq in label_freqs:
            for split, count in split_to_count.items():
                # minimum number of each class per split (int(x) truncates x)
                split_to_quota[split][label] = int(
                    (count / len(item_groups)) * freq)
        split_to_quota = dict(split_to_quota)

        self.metadata.update({
            'split.label_counts': dict(Counter(labels)),
            'split.label_split_quotas': split_to_quota
        })

        return split_to_quota

    @ex.capture
    def _shuffle(self, attrs_df, strata_key, item_key, item_label, _log):
        """
        """
        strata_groups = attrs_df.groupby(strata_key)
        strata_id_to_items = defaultdict(list)
        for strata_id, strata_group in strata_groups:
            item_groups = strata_group.groupby(item_key)
            for item_id, item_group in item_groups:
                label = item_group.loc[item_id][item_label]
                try:
                    label = list(set(label))[0]
                except:
                    _log.warning(f'item {item_id} only has one instance')
                strata_id_to_items[strata_id].append({
                    'item_id': item_id,
                    'label': label
                })

        strata_id_to_items = list(strata_id_to_items.items())
        return strata_id_to_items

    @ex.capture
    def _assign(self, strata_id_to_items, split_to_quota, split_to_count, train_split, _log):
        """
        Caution: splitter relies on randomness to hit quotas and will generate
        new splits until
        """
        while True:
            result = self._get_candidate(strata_id_to_items, split_to_quota)
            if result == None:
                continue
            split_to_label_freqs, split_to_item_ids = result
            _log.info(
                f'verifying candidate with label freqs: \n{split_to_label_freqs}')
            if self._verify_candidate(split_to_label_freqs, split_to_quota):
                _log.info(
                    f'verification success with quota: \n{split_to_quota}')
                break

        return split_to_item_ids

    @ex.capture
    def _get_candidate(self, strata_id_to_items, split_to_quota, split_to_count):
        """
        """
        random.shuffle(strata_id_to_items)
        split_to_item_ids = defaultdict(list)
        split_to_label_freqs = {split: {class_type: 0
                                        for class_type in split_to_quota[split]}
                                for split in split_to_quota}
        i = 0
        for split, count in split_to_count.items():
            added = 0
            while added < count:
                if i == len(strata_id_to_items):
                    return None
                strata_id, items = strata_id_to_items[i]
                for item in items:
                    item_id = item['item_id']
                    label = item['label']
                    split_to_item_ids[split].append(item_id)
                    split_to_label_freqs[split][label] += 1
                added += len(items)
                i += 1
        return split_to_label_freqs, split_to_item_ids

    @ex.capture
    def _verify_candidate(self, split_to_label_freqs, split_to_quota, train_split):
        """
        """
        for split, quota in split_to_quota.items():
            if split == train_split:
                continue
            for class_type, quota_count in quota.items():
                if split_to_label_freqs[split][class_type] < quota_count:
                    return False
        return True

    @ex.capture
    def _format(self, attrs_df, split_to_item_ids, item_key, item_label):
        """
        """
        split_to_split_df = {}
        split_to_labels = defaultdict(list)
        for split, item_ids in split_to_item_ids.items():
            split_df = attrs_df.loc[item_ids]
            split_to_split_df[split] = split_df

            item_groups = split_df.groupby(item_key)
            for item_id, item_group in item_groups:
                split_to_labels[split].append(
                    list(set(item_group[item_label]))[0])

        split_to_counts = defaultdict(dict)
        for split, labels in split_to_labels.items():
            counts = sorted(Counter(labels).items())
            for label, count in counts:
                split_to_counts[split][label] = count

        self.metadata.update({
            'split.label_to_freq': dict(split_to_counts)
        })
        return split_to_split_df


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    else:
        util.require_dir(config['exp_dir'])
    ex.observers.append(FileStorageObserver(config['exp_dir']))


@ex.main
def main(_run):
    """
    """
    splitter = DataSplitter()
    results = splitter.run()
    return results


if __name__ == '__main__':
    ex.run_commandline()
