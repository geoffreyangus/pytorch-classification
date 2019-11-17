from sacred import Ingredient

transforms_ingredient = Ingredient('transforms')


@transforms_ingredient.config
def config():
    preprocessing = {
        'x1': [
            {
                'class_name': 'GrayScale',
                'args': {
                    'num_output_channels': 1
                }
            }
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
            }
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
        'joint': []
    }
    # TODO: apply data augmentation to joint images
    augmentation = {
        'x1': [],
        'x2': [],
        'joint': []
    }
