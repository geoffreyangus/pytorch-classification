import sys
sys.path.append('..')

from types import SimpleNamespace 

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import models.cifar as models
from dataset import CIFAR100, collate_train, collate_test


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
    
def get_cnn(args, num_classes):
    """
    Loads CNN architecture in style of pytorch-classification
    """
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
        
    return torch.nn.DataParallel(model)
        
def load_trained_model(path):
    """
    Loads model trained with pytorch-classification
    
    TODO: Expand from resnext to other model types 
    """
    model_type = path.split('/')[-1].split('-')[0]
    superclass = 'superclass' in path
    if 'cifar100' in path:
        dataset = 'cifar100'
    # Right now, for resnext only
    args = {'arch': model_type, 'depth':29, 'cardinality':8, 'widen_factor': 4, 'drop': 0,
            'superclass': superclass, 'dataset':dataset, 
            'train_batch':128, 'test_batch':32, 'workers':6}
    args = SimpleNamespace(**args)
    num_classes = 20 if args.superclass else 100
    model = get_cnn(args, num_classes)
    checkpoint = torch.load(f'{path}/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    return model, args
    
def fetch_dataloaders(args, subsample_subclass={}, whiten_subclass={}, diff_subclass={}):
    """
    Preparing dataloaders 
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = CIFAR100
        num_classes = 20 if args.superclass > 0 else 100

    print(f'Using {num_classes} classes...')

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train, superclass=args.superclass,
                          subsample_subclass=subsample_subclass, whiten_subclass=whiten_subclass,
                          diff_subclass=diff_subclass)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, collate_fn=collate_train)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test, superclass=args.superclass,
                         subsample_subclass=subsample_subclass,whiten_subclass=whiten_subclass,
                         diff_subclass=diff_subclass)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, collate_fn=collate_test)
    
    return {'train':trainloader, 'test':testloader}