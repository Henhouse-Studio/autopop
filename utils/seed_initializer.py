import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the seed for reproducibility in various libraries.

    :param seed: The seed value to use. Default is 42 (int).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using GPU:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
