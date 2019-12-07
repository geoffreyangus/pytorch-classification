import numpy as np
from sacred import Ingredient

transforms_ingredient = Ingredient('transforms')


@transforms_ingredient.config
def config():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
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
                'class_name': 'ToRGB',
                'args': {}
            },
            {
                'class_name': 'Resize',
                'args': {
                    'size': 480
                }
            },
            {
                'class_name': 'CenterCrop',
                'args': {
                    'size': 480
                }
            },
            {
                'class_name': 'ToTensor',
                'args': {}
            },
        ],
        'x2': [
#             {
#                 'class_name': 'ChooseChannels',
#                 'args': {
#                     'channels': [1, 2]
#                 }
#             },
            {
                'class_name': 'ToTensor',
                'args': {}
            },
        ],
        'joint': [
            {
                'class_name': 'Normalize',
                'args': {
                    'mean': mean,
                    'std': std,
                }
            }
        ]
    }

    # TODO: apply data augmentation to joint images
    augmentation = {
        'x1': [],
        'x2': [],
        'joint': [
            {
                'class_name': 'ToPILImage',
                'args': {}
            },
#            {
#                'class_name': 'RandomAffine',
#                'args': {
#                    'degrees': 60,
#                    'translate': (0.1, 0.1),
#                    'scale': (0.75, 1.25),
#                }
#            },
            {
                'class_name': 'RandomHorizontalFlip',
                'args': {}
            },
            {
                'class_name': 'ToTensor',
                'args': {}
            }
        ]
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

    
class ToRGB:
    """
    Converts PIL image to 3-channel RGB.
    
    Returns a PIL Image.
    """
    def __call__(self, img):
        return img.convert('RGB')
