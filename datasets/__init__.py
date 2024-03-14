from .utils.continual_dataset import ContinualDataset
from argparse import Namespace
from .seq_cifar10 import SequentialCIFAR10
from .seq_cifar100 import SequentialCIFAR100
from .seq_imgnetsub import SequentialImageNetAnimals
from .seq_imagenet_r import SequentialImageNetR
from .label2name import LabelConverter
from .seq_cifar10 import MyCIFAR10
from .seq_cifar100 import MyCIFAR100
from .seq_imagenet_r import MyImageNetRDataset

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialImageNetAnimals.NAME: SequentialImageNetAnimals,
    SequentialImageNetR.NAME: SequentialImageNetR,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    return NAMES[args.dataset](args)

def get_test_datasets(args, transform):
    if args.dataset == 'seq-cifar10':
        return MyCIFAR10('/home/bqqi/lifelong_research/src/CL_Transformer/data/CIFAR10', train=False, download=True, transform=transform)
    elif args.dataset == 'seq-cifar100':
        return MyCIFAR100('/home/bqqi/lifelong_research/src/CL_Transformer/data/CIFAR100', train=False, download=True, transform=transform)
    elif args.dataset == 'seq-imagenet-r':
        return MyImageNetRDataset(is_train=False)