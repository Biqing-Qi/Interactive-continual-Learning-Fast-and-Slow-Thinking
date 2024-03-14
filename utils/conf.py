import random
import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())


def get_device(args) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return "./data/"


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
