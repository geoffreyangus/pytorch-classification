import sys
sys.path.append('..')

from types import SimpleNamespace 

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score
from collections import defaultdict

import models.cifar as models
from dataset import CIFAR100, collate_train, collate_test

def get_coarse_accuracies(prediction_df):
# Computing superclass accuracy and subclass accuracy
    coarse_results = defaultdict(dict)
    accuracy_type = 'coarse'
    for coarse_class in prediction_df['coarse_labels_string'].unique():
        coarse_accs = {}
        coarse_class_df = prediction_df[prediction_df['coarse_labels_string']==coarse_class]
        coarse_class_acc = accuracy_score(coarse_class_df[f'{accuracy_type}_labels'],coarse_class_df[f'preds'])
        print(f'{accuracy_type} label accuracy: {coarse_class_acc} \t (superclass: {coarse_class})')
        print('--')
        coarse_results['superclass'].update({coarse_class:coarse_class_acc})
       # print(f"fine label disribution: {Counter(coarse_class_df['fine_labels_string'])}")
        for ii, fine_class in enumerate(coarse_class_df['fine_labels_string'].unique()):
            fine_class_df = coarse_class_df[coarse_class_df['fine_labels_string']==fine_class]
            fine_class_acc = accuracy_score(fine_class_df[f'{accuracy_type}_labels'],fine_class_df[f'preds'])
            print(f'{accuracy_type} label accuracy: {fine_class_acc}\t (subclass: {fine_class})')
            coarse_results[f'subclass_{ii}'].update({coarse_class:fine_class_acc})
        print('==\n')
    return coarse_results
    
def extract_resnext_features(md,x):
    """
    Extract resnext features
    """
    x = md.module.conv_1_3x3.forward(x)
    x = F.relu(md.module.bn_1.forward(x), inplace=True)
    x = md.module.stage_1.forward(x)
    x = md.module.stage_2.forward(x)
    x = md.module.stage_3.forward(x)
    x = F.avg_pool2d(x, 8, 1)
    x = x.view(-1, 1024)
    return x
    
def get_cnn(model_name, model_args):
    """
    Loads CNN architecture in style of pytorch-classification
    """
    model = models.__dict__[model_name](
        **model_args
    )        
    return torch.nn.DataParallel(model)
        
def load_trained_model(cifar_type, model_name, model_args, checkpoint_dir, lmcl_args=None):
    
    model = get_cnn(model_name, model_args)
    checkpoint = torch.load(f'{checkpoint_dir}/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    if lmcl_args != None:
        criterion = getattr(losses, 'LMCL_loss')(**lmcl_args)
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        return model, criterion
    return model
    
def fetch_dataloaders(data_dir, cifar_type, superclass, dataset_configs, dataloader_configs):
    """
    Preparing dataloaders 
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_configs['train']['transform'] = transform_train

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_configs['test']['transform'] = transform_test
    
    if cifar_type == 'CIFAR10':
        dataset_class = datasets.CIFAR10
        num_classes = 10
    else:
        dataset_class = CIFAR100
        num_classes = 20 if superclass else 100

    print(f'Using {num_classes} classes...')

    datasets = {}
    for split, ds_args in dataset_configs.items():
        datasets[split] = dataset_class(root=data_dir, train=(split=='train'), download=False,
                                        **ds_args)
    
    dataloaders = {}
    for split, dl_args in dataloader_configs.items():
        dataloaders[split] = data.DataLoader(datasets[split], 
                                             collate_fn=collate_train if split=='train' else collate_test,
                                             **dl_args)

    return dataloaders