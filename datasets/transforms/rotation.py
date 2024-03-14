import numpy as np
import torchvision.transforms.functional as F


class Rotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min: int = 0, deg_max: int = 180) -> None:  # 30  80
        """
        Initializes the rotation with a random angle.
        :param deg_min: lower extreme of the possible random angle
        :param deg_max: upper extreme of the possible random angle
        """
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        return F.rotate(x, self.degrees)


class FixedRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, seed: int, deg_min: int = 0, deg_max: int = 180) -> None:
        """
        Initializes the rotation with a random angle.
        :param seed: seed of the rotation
        :param deg_min: lower extreme of the possible random angle
        :param deg_max: upper extreme of the possible random angle
        """
        self.seed = seed
        self.deg_min = deg_min
        self.deg_max = deg_max

        np.random.seed(seed)
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        return F.rotate(x, self.degrees)


class IncrementalRotation(object):
    """
    Defines an incremental rotation for a numpy array.
    """

    def __init__(
        self, init_deg: int = 0, increase_per_iteration: float = 0.006
    ) -> None:
        """
        Defines the initial angle as well as the increase for each rotation
        :param init_deg:
        :param increase_per_iteration:
        """
        self.increase_per_iteration = increase_per_iteration
        self.iteration = 0
        self.degrees = init_deg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        degs = (self.iteration * self.increase_per_iteration + self.degrees) % 360
        self.iteration += 1
        return F.rotate(x, degs)

    def set_iteration(self, x: int) -> None:
        """
        Set the iteration to a given integer
        :param x: iteration index
        """
        self.iteration = x
