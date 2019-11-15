import random

from pypadre.core.util.optional import pytorch_exists

padre_seed = None


def set_seeds(seed):
    global padre_seed
    padre_seed = seed
    random.seed(seed)

    # set numpy seed
    import numpy
    numpy.random.seed(seed)
    # global seeds for numpy seem to not work with RandomState()

    # set pytorch seed
    if pytorch_exists:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
