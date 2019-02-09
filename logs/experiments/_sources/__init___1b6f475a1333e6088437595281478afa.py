from .BN_Inception import BN_Inception
from .FineTune import Fine_Tune

__factory = {
    'BN-Inception': BN_Inception,
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name in __factory:
        return __factory[name](*args, **kwargs)
    else:
        return Fine_Tune(name, *args, **kwargs)
