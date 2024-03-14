import sys
import os

sys.path.append(os.getcwd())


def create_if_not_exists(path: object) -> object:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def apply_decay(decay, lr, optimizer, num_iter):
    if decay != 1:
        learn_rate = lr * (decay**num_iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learn_rate
