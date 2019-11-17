import numpy as np
from sacred import Ingredient

transforms_ingredient = Ingredient('transforms')


@transforms_ingredient.config
def config():
    preprocessing = {
        'x1': [
#             {
#                 'class_name': 'Grayscale',
#                 'args': {
#                     'num_output_channels': 1
#                 }
#             },
#             {
#                 'class_name': 'Unsqueeze',
#                 'args': {
#                     'axis': -1
#                 }
#             },
            {
                'class_name': 'ToTensor',
                'args': {}
            },
            {
                'class_name': 'Normalize',
                'args': {
                    'mean': [0.5],
                    'std': [0.5],
                }
            }
        ],
        'x2': [
            {
                'class_name': 'ChooseChannels',
                'args': {
                    'channels': [1, 2]
                }
            },
            {
                'class_name': 'ToTensor',
                'args': {}
            },
            {
                'class_name': 'Normalize',
                'args': {
                    'mean': [0.5, 0.5],
                    'std': [0.5, 0.5],
                }
            }
        ],
        'joint': [
            {
                'class_name': 'ToPILImage',
                'args': {}
            },
            {
                'class_name': 'Resize',
                'args': {
                    'size': 224
                }
            },
            {
                'class_name': 'ToTensor',
                'args': {}
            }
        ]
    }

    # TODO: apply data augmentation to joint images
    augmentation = {
        'x1': [],
        'x2': [],
        'joint': []
    }


class ChooseChannels:
    """
    Select a subset of available channels in a PIL Image.

    Returns a numpy array.
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        return np.array(img)[:, :, self.channels]

    
class Unsqueeze:
    """
    Unsqueezes a numpy array at the given axis

    Returns a numpy array.
    """
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, img):
        return np.expand_dims(img, axis=self.axis)
