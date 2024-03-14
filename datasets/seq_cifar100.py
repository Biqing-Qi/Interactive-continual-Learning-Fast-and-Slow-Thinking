from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import sys, os

sys.path.append(os.getcwd())
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from .utils.validation import get_train_val
from backbone.CCT_our import CVT, Brain_Coworker, Brain_Coworker_Vit, VitPre
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
    RandomRotation
)
import torch.nn as nn


class MyCIFAR100(CIFAR100):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.not_aug_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):
    image_size = 224
    channel = 3

    NAME = "seq-cifar100"
    SETTING = "class-il"
    N_CLASSES_TOTAL = 100
    N_TASKS = 10  # 10
    N_CLASSES_PER_TASK = int(N_CLASSES_TOTAL / N_TASKS)

    def __init__(self, args) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(SequentialCIFAR100, self).__init__(args)
        self.hidden_dim = args.hidden_dim
        if self.args.use_albumentations:
            self.TRANSFORM = A.Compose(
                [
                    A.CropAndPad(px=4),
                    A.RandomCrop(width=224, height=224),
                    A.HorizontalFlip(),
                    A.Normalize(
                        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]
                    ),
                    ToTensorV2(),
                ]
            )
            self.test_transform = A.Compose(
                [ToTensorV2(), self.get_normalization_transform()]
            )

        else:
            self.TRANSFORM = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                    #    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    #),
                ]
            )
            self.test_transform = transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor(), 
                 #transforms.Normalize(
                 #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

            self.TRANSFORM_SC = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            #RandomHorizontalFlip(),
            RandomRotation(degrees=30),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.2),
            #transforms.Normalize(
            #            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.Normalize(
            #            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            )

    def get_data_loaders(self, nomask=False):
        transform = self.TRANSFORM

        test_transform = self.test_transform

        train_dataset = MyCIFAR100(
            base_path() + "CIFAR100", train=True, download=True, transform=transform
        )
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, test_transform, self.NAME
            )
        else:
            test_dataset = CIFAR100(
                base_path() + "CIFAR100",
                train=False,
                download=True,
                transform=test_transform,
            )

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
                train_dataset, batch_size=32, shuffle=True, num_workers=4
            )
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )

            return train_loader, test_loader

    def get_joint_loaders(self, nomask=False):
        return self.get_data_loaders(nomask=True)

    def not_aug_dataloader(self, batch_size):
        transform = self.test_transform

        train_dataset = MyCIFAR100(
            base_path() + "CIFAR100", train=True, download=True, transform=transform
        )

        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    def get_transform(self):
        if self.args.use_albumentations:
            transform = self.TRANSFORM
        else:
            transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])
        return transform

    # @staticmethod
    # def get_transform():
    #     transform = transforms.Compose([transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
    #     return transform

    @staticmethod
    def get_backbone():
        output_dim = SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS
        return resnet18(output_dim)

    @staticmethod
    def get_backbone_cct():
        output_dim = SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS
        return CVT(32, output_dim)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_normalization_transform(self):
        if self.args.use_albumentations:
            transform = A.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2673, 0.2564, 0.2761],
            )
        else:
            transform = transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
            )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        return transform
    
    @staticmethod
    def get_backbone_brain_coworker():
        output_dim = SequentialCIFAR100.N_CLASSES_PER_TASK*SequentialCIFAR100.N_TASKS
        return Brain_Coworker(128, output_dim)

    @staticmethod
    def get_backbone_brain_coworker_vit(args):
        output_dim = SequentialCIFAR100.N_CLASSES_PER_TASK*SequentialCIFAR100.N_TASKS
        return Brain_Coworker_Vit(128, output_dim, args.hidden_dim, SequentialCIFAR100.N_CLASSES_PER_TASK)
    
    @staticmethod
    def get_backbone_vit_pre(args):
        output_dim = SequentialCIFAR100.N_CLASSES_PER_TASK*SequentialCIFAR100.N_TASKS
        return VitPre(output_dim)