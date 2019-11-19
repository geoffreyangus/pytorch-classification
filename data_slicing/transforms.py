import numpy as np
from sacred import Ingredient

transforms_ingredient = Ingredient('transforms')


@transforms_ingredient.config
def config():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocessing = [
        {
            'class_name': 'Resize',
            'args': {
                'size': 224
            }
        },
        {
            'class_name': 'CenterCrop',
            'args': {
                'size': 224
            }
        },
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
