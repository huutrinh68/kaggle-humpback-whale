from .Whale import Whale
# from .transforms import *
import os

__factory = {
    'whale': Whale
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, mode='train', *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))

    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, mode=mode, *args, **kwargs)
