from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from typing import Tuple
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import torchvision.transforms.functional as transofrms_f
from .multi_dataloader import CudaDataLoader, MultiEpochsDataLoader
import sys
import os
sys.path.append(os.getcwd())
from utils.conf import get_device


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """

    NAME = None
    image_size = None
    channel = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders_minist(
    train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )
    test_mask = np.logical_and(
        np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    current_task_id = int(setting.i / setting.N_CLASSES_PER_TASK + 1)

    if setting.args.model == "our":
        batch_size = int(setting.args.batch_size / current_task_id + 2)
        #  2 * (batch_size-4) 在500, 200的时候更好
        idx = torch.randint(
            len(train_dataset.data),
            (
                int(
                    len(train_dataset.data)
                    / current_task_id
                    * (2 * (batch_size) / (batch_size - 4))
                ),
            ),
        )
    else:
        batch_size = setting.args.batch_size

    if setting.args.buffer_size != 5120 and setting.args.model == "our":
        train_dataset.data = train_dataset.data[idx]
        train_dataset.targets = train_dataset.targets[idx]  # 5120内存时要去掉随机抽样,改用全量数据
    print("each task data num: ", len(train_dataset.data))
    print("loader batch_size: ", batch_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4
    )
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK  # 增加了task

    return train_loader, test_loader


def store_masked_loaders_core50(
    train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # core50 (i.e. uneven # of classes)
    if type(setting.N_CLASSES_PER_TASK_list) == list:
        FROM_CLASS = np.sum(setting.N_CLASSES_PER_TASK_list[: setting.i])
        TO_CLASS = np.sum(setting.N_CLASSES_PER_TASK_list[: setting.i + 1])
    # any other dataset
    else:
        FROM_CLASS = setting.i * setting.N_CLASSES_PER_TASK
        TO_CLASS = (setting.i + 1) * setting.N_CLASSES_PER_TASK

    train_mask = np.logical_and(
        np.array(train_dataset.targets % 1000) >= FROM_CLASS,
        np.array(train_dataset.targets % 1000) < TO_CLASS,
    )
    test_mask = np.logical_and(
        np.array(test_dataset.targets % 1000) >= FROM_CLASS,
        np.array(test_dataset.targets % 1000) < TO_CLASS,
    )

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4
    )
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader


def store_masked_loaders(
    train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )
    test_mask = np.logical_and(
        np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    print("each task data num: ", len(train_dataset.data))

    if setting.args.use_distributed:
        print("use DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    num_workers = 8
    pin_memory = False

    cuda_dataloader = False
    if cuda_dataloader:
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=setting.args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory,
        )

        test_loader = MultiEpochsDataLoader(
            test_dataset,
            batch_size=setting.args.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=test_sampler,
            pin_memory=pin_memory,
        )
        if torch.cuda.is_available():
            train_loader = CudaDataLoader(train_loader, get_device(setting.args))
            test_loader = CudaDataLoader(test_loader, get_device(setting.args))
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=setting.args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=setting.args.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=test_sampler,
            pin_memory=pin_memory,
        )
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK  # 增加了task

    return train_loader, test_loader


def masked_all_loaders_train(
    train_dataset: datasets, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= 0,
        np.array(train_dataset.targets) < setting.N_CLASSES_PER_TASK * setting.N_TASKS,
    )

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    print("all task data num: ", len(train_dataset.data))

    if setting.args.use_distributed:
        print("use DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    num_workers = 8
    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=setting.args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
    )
    setting.train_loader = train_loader

    return train_loader


def get_previous_train_loader(
    train_dataset: datasets, batch_size: int, setting: ContinualDataset
) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """

    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK,
        np.array(train_dataset.targets) < setting.i,
    )

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_previous_gan_loader(
    train_dataset: datasets, batch_size: int, setting: ContinualDataset
):
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK,
        np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK,
    )

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    train_x1 = []
    train_x2 = []
    train_x1_label = []
    for label_id in set(train_dataset.targets):
        idx = train_dataset.targets == label_id
        data = train_dataset.data[idx]
        labels = train_dataset.targets[idx]
        print("sample_num_per_class: ", data.shape)
        x2_data = list(data)
        np.random.shuffle(x2_data)
        train_x1.extend(data)
        train_x2.extend(x2_data)
        train_x1_label.extend(labels)

        # data2 =  train_dataset.data[idx]
        # train_x1.extend(data2)
        # train_x2.extend(data2)
        # train_x2.extend(data)

    in_channels = train_dataset.data.shape[-1]
    img_size = train_dataset.data.shape[2]

    gan_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            setting.get_normalization_transform(),
        ]
    )

    train_dataset = DaganDataset(
        train_x1,
        train_x1_label,
        train_x2,
        gan_transform,
        setting.get_denormalization_transform(),
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# all data !!!!!!!
def get_all_gan_loader(
    train_dataset: datasets, batch_size: int, setting: ContinualDataset
):
    aug_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            # transforms.RandomHorizontalFlip(),
            # AugmentRotation(),
            transforms.ToTensor(),
        ]
    )

    train_dataset.targets = np.array(train_dataset.targets)
    train_x1 = []
    train_x2 = []
    train_x1_label = []
    for label_id in set(train_dataset.targets):
        idx = train_dataset.targets == label_id
        data = train_dataset.data[idx]
        labels = train_dataset.targets[idx]
        # print('sample_num_per_class: ', data.shape)  sample_num_per_class: (5000, 32, 32, 3) for cifar10
        x2_data = list(data)
        np.random.shuffle(x2_data)

        # x2_data = [aug_transforms(i) for i in data]  # .transpose_(1, 2) # imageNet !!
        data = [np.uint8(255 * i) for i in data]  # imageNet !!
        # x2_data = [i for i in data]

        train_x1.extend(data)
        train_x2.extend(x2_data)
        train_x1_label.extend(labels)

        # train_x2.extend(aug_transforms(data))

        # train_x1.extend(data)
        # train_x2.extend(data)
        # train_x1_label.extend(labels)

    in_channels = train_dataset.data.shape[-1]
    img_size = train_dataset.data.shape[2]

    gan_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            setting.get_normalization_transform(),
        ]
    )

    train_dataset = DaganDataset(
        train_x1,
        train_x1_label,
        train_x2,
        gan_transform,
        setting.get_denormalization_transform(),
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_buffer_loaders_gan(buf_x, buf_y, batch_size, setting):
    train_x1 = []
    train_x2 = []
    train_x1_label = []
    for label_id in set(buf_y):
        idx = buf_y == label_id
        data = buf_x[idx]
        print("sample_num_per_class: ", data.shape)
        labels = buf_y[idx]
        x2_data = list(data)
        np.random.shuffle(x2_data)
        train_x1.extend(data)
        train_x2.extend(x2_data)
        train_x1_label.extend(labels)

        # data2 =  train_dataset.data[idx]
        # train_x1.extend(data2)
        # train_x2.extend(data2)
        # train_x2.extend(data)

    in_channels = buf_x.shape[-1]
    img_size = buf_x.shape[2]

    gan_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            setting.get_normalization_transform(),
        ]
    )

    train_dataset = DaganDataset(
        train_x1,
        train_x1_label,
        train_x2,
        gan_transform,
        setting.get_denormalization_transform(),
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


class DaganDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1_examples, x1_labels, x2_examples, transform, denormalization):
        assert len(x1_examples) == len(x2_examples)
        self.x1_examples = x1_examples
        self.x1_labels = x1_labels
        self.x2_examples = x2_examples
        self.transform = transform
        self.denormalization = denormalization

    def __len__(self):
        return len(self.x1_examples)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.x1_examples[idx]), self.transform(
                self.x2_examples[idx]
            )


class AugmentRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min: int = 90, deg_max: int = 90) -> None:  # 设为0,结果就不会再随机了
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
        return transofrms_f.rotate(x, self.degrees)
