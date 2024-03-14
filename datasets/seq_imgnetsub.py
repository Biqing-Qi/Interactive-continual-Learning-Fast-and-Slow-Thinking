from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import sys, os

sys.path.append(os.getcwd())

import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from .utils.validation import get_train_val
from backbone.CCT_our import CVT, Brain_Coworker, Brain_Coworker_Vit
from .utils.continual_dataset import ContinualDataset, store_masked_loaders
from .utils.continual_dataset import get_previous_train_loader
from .transforms.denormalization import DeNormalize
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from typing import Tuple
import numpy as np
import torchvision.models as models
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomGaussianBlur,
    Normalize
)
import torch.nn as nn
import os

from collections import OrderedDict
from torchvision.datasets import VisionDataset
from typing import Callable, Any
from argparse import Namespace


class MyImageNetAnimals(VisionDataset):
    def __init__(
        self,
        root: str = "/home/bqqi/lifelong_research/workspace/lifelong_data",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        train=True,
        return2term=False,
    ) -> None:
        super().__init__(root, None, transform, target_transform)

        self.not_aug_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ],
        )
        classes = OrderedDict.fromkeys(
            ["fish", "bird", "snake", "dog", "butterfly", "insect"]
        )

        for class_ in classes.keys():
            classes[class_] = os.listdir(os.path.join(root, "train", class_))
        all_subclass = sum(classes.values(), [])
        subclass2id = {subclass: i for i, subclass in enumerate(all_subclass)}
        self._classes = classes
        self._subclass2id = subclass2id
        samples = []
        for class_, subclasses_ in classes.items():
            for subclass_ in subclasses_:
                if train:
                    for img in os.listdir(os.path.join(root, "train", class_, subclass_)):
                        samples.append(
                            (
                                os.path.join(root, "train", class_, subclass_, img),
                                subclass2id[subclass_],
                            )
                        )
                else:
                    for img in os.listdir(os.path.join(root, "test", class_, subclass_)):
                        samples.append(
                            (
                                os.path.join(root, "test", class_, subclass_, img),
                                subclass2id[subclass_],
                            )
                        )
        self.targets = [s[1] for s in samples]
        self.data = [s[0] for s in samples]
        self.data = np.array(self.data)
        self.return2term = return2term


    def __getitem__(self, index: int) -> Any:
        img = self.data[index]
        img = Image.open(img).convert("RGB")
        target = self.targets[index]
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]
        if self.return2term:
            return img, target
        return img, target, not_aug_img

    def __len__(self) -> int:
        return len(self.targets)


class SequentialImageNetAnimals(ContinualDataset):
    image_size = 224
    channel = 3

    NAME = "seq-imagenet-animals"
    SETTING = "class-il"
    N_CLASSES_TOTAL = 36
    N_TASKS = 6
    N_CLASSES_PER_TASK = 6

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        super(SequentialImageNetAnimals, self).__init__(args)
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.TRANSFORM = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], 
                #                  [0.5, 0.5, 0.5])
                #self.get_normalization_transform(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], 
                #                  [0.5, 0.5, 0.5])
                #self.get_normalization_transform(),
            ]
        )

        self.TRANSFORM_SC = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.2),
            #transforms.Normalize([0.5, 0.5, 0.5], 
            #                      [0.5, 0.5, 0.5])
            #RandomGaussianBlur((3,3),(0.1,2.0), p=1.0),
            #Normalize(torch.FloatTensor((0.485, 0.456, 0.406)), torch.FloatTensor((0.229, 0.224, 0.225)))
        )


    def get_data_loaders(self, nomask=False) -> Tuple[DataLoader, DataLoader]:
        transform = self.TRANSFORM
        test_transform = self.test_transform

        train_dataset = MyImageNetAnimals(transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, test_transform, self.NAME
            )
        else:
            # raise NotImplementedError("还没有实现")
            print("\033[1;33m Warning: not using validation\033[0m")
            from copy import deepcopy

            test_dataset = MyImageNetAnimals(transform=test_transform, train=False, return2term=True)

        if not nomask:
            if isinstance(train_dataset.targets, list):
                train_dataset.targets = torch.tensor(
                    train_dataset.targets, dtype=torch.long
                )
            if isinstance(test_dataset.targets, list):
                test_dataset.targets = torch.tensor(
                    test_dataset.targets, dtype=torch.long
                )
            train, test = store_masked_loaders(train_dataset, test_dataset, self)
            return train, test
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            return train_loader, test_loader

    def get_joint_loaders(self, nomask=False):
        return self.get_data_loaders(nomask=True)

    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        raise NotImplementedError("还没实现")

    def get_normalization_transform(self):
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_denormalization_transform(self):
        return DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_backbone():
        output_dim = SequentialImageNetAnimals.N_CLASSES_TOTAL
        return resnet18(output_dim)

    @staticmethod
    def get_backbone_cct():
        output_dim = SequentialImageNetAnimals.N_CLASSES_TOTAL
        return CVT(128, output_dim)

    @staticmethod
    def get_backbone_brain_coworker():
        output_dim = SequentialImageNetAnimals.N_CLASSES_TOTAL
        return Brain_Coworker(128, output_dim)

    @staticmethod
    def get_backbone_brain_coworker_vit():
        output_dim = SequentialImageNetAnimals.N_CLASSES_TOTAL
        return Brain_Coworker_Vit(128, output_dim, SequentialImageNetAnimals.N_CLASSES_PER_TASK)