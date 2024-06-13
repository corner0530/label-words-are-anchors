import os
import random

import numpy as np
import torch.backends.cudnn


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def np_temp_random(
    seed,
):
    """
    Decorator that temporarily sets the seed for NumPy's random number generator.

    Args:
        seed (int): The seed value to set for NumPy's random number generator.

    Returns:
        callable: A decorator function that can be used to wrap other functions.
    """

    def np_temp_random_inner(func):
        def np_temp_random_inner_inner(*args, **kwargs):
            ori_state = np.random.get_state()
            np.random.seed(seed)
            result = func(*args, **kwargs)
            np.random.set_state(ori_state)
            return result

        return np_temp_random_inner_inner

    return np_temp_random_inner
