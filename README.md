# Interactive Continual Learning (ICL)

Code for CVPR 2024 paper [Interactive Continual Learning: Fast and Slow Thinking](https://arxiv.org/pdf/2403.02628.pdf)

## Dependencies

```shell
pip install -r requirements.txt
```

## Dataset Preparation

1. For CIFAR10 and CIFAR100 datasets, the script automatically downloads.

2. For ImageNet-R dataset refer to the following link: https://github.com/hendrycks/imagenet-r

## Prepare for Slow System

1. For MiniGPT4 refer to the following github link: https://github.com/Vision-CAIR/MiniGPT-4

2. For INF-MLLM refer to the following github link: https://github.com/infly-ai/INF-MLLM

3. For PureMM refer to the following github link: https://github.com/Q-MM/PureMM

Download the github repository of MLLM to utils, and download the pre-training weight file to the specified file.

## Quick Start

Train and evaluate models through `utils/main.py`. For example, to train our model on Split CIFAR-10 with 500 fixed-size buffers, and include PureMM as System 2 in the test of the last task, one would execute:

```shell
python utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM
```

To compare training results without System 2, simply run:
```shell
python utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5
```

More datasets and methods are supported. You can find the available options by running:
```shell
python utils/main.py --help
```