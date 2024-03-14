# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
from torchvision.datasets import VisionDataset
from typing import Callable, Any
from argparse import Namespace
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomGaussianBlur,
    RandomRotation,
    Normalize
)
import torch
import torch.nn as nn
from backbone.CCT_our import CVT, Brain_Coworker, Brain_Coworker_Vit, VitPre

class MyImageNetRDataset(Dataset):
    def __init__(self,
                 root: str = "/home/bqqi/lifelong_research/workspace/lifelong_data/imagenet-r",
                #  transform: Callable[..., Any] | None = None,
                 is_train = True,
                 return2term = False
                 ) -> None:
        super(MyImageNetRDataset).__init__()
        
        self.not_aug_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], 
                #                  [0.5, 0.5, 0.5])
            ],
        )
        
        # self.transform = transform
        self.root = root
        self.is_train = is_train
        self.return2term = return2term
        self.class_dict = self.get_class_dict()
        self.train_img_names = []
        self.test_img_names = []
        self.train_label_id = []
        self.test_label_id = []
        self.train_label_name = []
        self.test_label_name = []
        self.label2name = {}
        
        for id, kv in enumerate(self.class_dict.items()):
            key, value = kv
            train_, test_ = self.train_test_split(key, ratio=0.8)
            self.train_img_names += train_
            train_label_id = [id] * len(train_)
            self.train_label_id += train_label_id
            train_label_name = [value] * len(train_)
            self.train_label_name += train_label_name
            self.test_img_names += test_
            test_label_id = [id] * len(test_)
            self.test_label_id += test_label_id
            test_label_name = [value] * len(test_)
            self.test_label_name += test_label_name
            self.label2name[str(id)] = value
            
        if is_train:
            self.data = self.train_img_names
            self.data = np.array(self.data)
            self.targets = self.train_label_id
            self.label_names = self.train_label_name
            print('number of train image:' + str(len(self.targets)))
            
        else:
            self.data = self.test_img_names
            self.data = np.array(self.data)
            self.targets = self.test_label_id
            self.label_names = self.test_label_name
            print('number of test images:' + str(len(self.targets)))
            
        

    def __getitem__(self, index) -> Any:
        img = self.data[index]
        img = Image.open(img).convert("RGB")
        target = self.targets[index]
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        img = self.not_aug_transform(img)

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]
        if self.return2term:
            return img, target
        return img, target, not_aug_img
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def get_class_dict(self):
        """
        返回ImageNet-R所有的类别形成一个字典
        """
        with open(os.path.join(self.root, 'class.txt'), 'r') as file:
            lines = file.readlines()
            
        class_dict = {}
        for line in lines:
            parts = line.split()
            class_dict[parts[0]] = parts[1]
            
        return class_dict
    
    def train_test_split(self, class_id, ratio=0.8):
        """
        这个函数主要是针对每个类，输出该类别的训练集和测试集
        class_id : 类别名称
        ratio : 训练集比例
        """
        img_name_list = os.listdir(os.path.join(self.root, class_id))
        split_index = int(len(img_name_list) * ratio)
        
        random.shuffle(img_name_list)
        
        train_img_name = img_name_list[:split_index]
        test_img_name = img_name_list[split_index:]
        
        train_img_name = [os.path.join(self.root, class_id, s) for s in train_img_name]
        test_img_name = [os.path.join(self.root, class_id, s) for s in test_img_name]
        
        return train_img_name, test_img_name


class SequentialImageNetR(ContinualDataset):
    image_size = 224
    
    NAME = 'seq-imagenet-r'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.Resize((image_size, image_size)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             #transforms.Normalize([0.5, 0.5, 0.5], 
             #                     [0.5, 0.5, 0.5])
             #transforms.Normalize([0.485, 0.456, 0.406], 
             #                     [0.229, 0.224, 0.225])
             ]
    )
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        super(SequentialImageNetR, self).__init__(args)
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
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
                 #                 [0.5, 0.5, 0.5])
                #self.get_normalization_transform(),
            ]
        )

        self.TRANSFORM_SC = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            RandomRotation(degrees=30),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.2),
            # RandomGaussianBlur((3,3),(0.1,2.0), p=1.0),
            #Normalize(torch.FloatTensor((0.5, 0.5, 0.5)), torch.FloatTensor((0.5, 0.5, 0.5)))
            #Normalize(torch.FloatTensor((0.485, 0.456, 0.406)), torch.FloatTensor((0.229, 0.224, 0.225)))
        )
    
    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize([0.5, 0.5, 0.5], 
             #                     [0.5, 0.5, 0.5])
             #self.get_normalization_transform()
            ]
        )

        train_dataset = MyImageNetRDataset()
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = MyImageNetRDataset(is_train=False)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(),])#, transforms.Normalize([0.5, 0.5, 0.5], 
                                  #[0.5, 0.5, 0.5])])#self.get_normalization_transform()])

        train_dataset = MyImageNetRDataset(transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImageNetR.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialImageNetR.N_CLASSES_PER_TASK
                        * SequentialImageNetR.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_backbone_cct():
        output_dim = SequentialImageNetR.N_CLASSES_PER_TASK*SequentialImageNetR.N_TASKS
        return CVT(128, output_dim)

    @staticmethod
    def get_backbone_brain_coworker():
        output_dim = SequentialImageNetR.N_CLASSES_PER_TASK*SequentialImageNetR.N_TASKS
        return Brain_Coworker(128, output_dim)

    @staticmethod
    def get_backbone_brain_coworker_vit(args):
        output_dim = SequentialImageNetR.N_CLASSES_PER_TASK*SequentialImageNetR.N_TASKS
        return Brain_Coworker_Vit(128, output_dim, args.hidden_dim, SequentialImageNetR.N_CLASSES_PER_TASK)
    
    @staticmethod
    def get_backbone_vit_pre(args):
        output_dim = SequentialImageNetR.N_CLASSES_PER_TASK*SequentialImageNetR.N_TASKS
        return VitPre(output_dim)