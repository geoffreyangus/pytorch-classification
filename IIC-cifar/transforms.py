import numpy as np
from sacred import Ingredient

transforms_ingredient = Ingredient('transforms')


@transforms_ingredient.config
def config():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocessing = [
#         {
#             'class_name': 'Resize',
#             'args': {
#                 'size': 64
#             }
#         },
#         {
#             'class_name': 'CenterCrop',
#             'args': {
#                 'size': 64
#             }
#         },
        {
            'class_name': 'ToTensor',
            'args': {}
        },
        {
            'class_name': 'Normalize',
            'args': {
                'mean': mean,
                'std': std
            }
        }
    ]

    augmentation = [
        {
            'class_name': 'RandomHorizontalFlip',
            'args': {}
        }
    ]

    g = [
        {
            'class_name': 'ToPILImage',
            'args': {}
        },
        {
            'class_name': 'RandomHorizontalFlip',
            'args': {}
        },
        {
            'class_name': 'RandomRotation',
            'args': {
                'degrees': 45
            }
        },
        {
            'class_name': 'ColorJitter',
            'args': {
                'brightness': 0.5,
                'contrast': 0.5,
                'saturation': 0.5,
                'hue': 0.0
            }
        },
        {
            'class_name': 'RandomGrayscale',
            'args': {
                'p': 0.5
            }
        }
    ]
