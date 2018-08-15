from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train
from .mobilenet1_train import Mobilenet_train
from .mobilenet1_test import Mobilenet_test
def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'Mobilenet':
        if name.split('_')[1] == 'test':
           return Mobilenet_test()
        elif name.split('_')[1] == 'train':
           return Mobilenet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
