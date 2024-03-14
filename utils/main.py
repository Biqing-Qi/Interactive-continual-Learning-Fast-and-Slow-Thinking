# coding=UTF-8
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import importlib

from datasets import NAMES as DATASET_NAMES


from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.training_diffusion import train as train_diffusion
from utils.best_args import best_args
from utils.conf import set_random_seed
import numpy as np
# from torchsummary import summary

import torch


import torchvision


import torch.distributed as dist
from confusion_matrix import plot_confusion
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(random.choice([1]))


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def main():
    parser = ArgumentParser(description="mammoth", allow_abbrev=False)
    parser.add_argument(
        "--model", type=str, required=True, help="Model name.", choices=get_all_models()
    )
    parser.add_argument(
        "--load_best_args",
        action="store_true",
        help="Loads the best arguments for each method, " "dataset and memory buffer.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="node rank for distributed training"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    # 增加一个新的参数 --ganmem
    parser.add_argument("--ganmem", default=False, action="store_true")
    parser.add_argument("--diffumem", default=False, action="store_true")
    parser.add_argument("--with_brain", default=False, action="store_true")
    #这里是我们的方法
    parser.add_argument("--with_brain_vit", default=False, action="store_true")
    #这个不用管
    parser.add_argument("--use_screening", default=False, action="store_true")
    #是否添加slow辅助
    parser.add_argument("--with_slow", default=False, action="store_true")
    parser.add_argument("--imsize", default=224, type=int)
    parser.add_argument("--num_classes", default=36, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", default=896, type=int)
    #第一个超参，集中度
    parser.add_argument("--kappa", default=1, type=float)
    #第二个超参，正则项系数
    parser.add_argument("--lmbda", default=0.1, type=float)
    #第三个超参，正则项计算的margin
    parser.add_argument("--delta", default=0.1, type=float)
    #第四个超参，Topk元素
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--cfg-path", default = '/home/bqqi/lifelong_research/src/CL_Transformer/utils/MiniGPT4/eval_configs/minigpt4_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model. Use -1 for CPU.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--vit_finetune", default=False, action="store_true")
    # for other slow models
    parser.add_argument("--slow_model", type=str, default='PureMM')
    parser.add_argument("--conv_mode", type=str, default='vicuna_v1')
    # for prue args
    parser.add_argument("--pure_model_path", type=str, default='/home/bqqi/ICL/utils/PureMM/model/PureMM_v1.0')
    parser.add_argument("--pure_model_base", type=str, default='/home/bqqi/ICL/utils/PureMM/model/vicuna-13b-v1.5')
    # for inf-mllm args
    parser.add_argument("--inf_model_path", type=str, default='/home/bqqi/ICL/utils/INF-MLLM/InfMLLM_13B_Chat')
    parser.add_argument("--inf_temperature", type=float, default=0)
    parser.add_argument("--inf_top_p", type=float, default=None)
    parser.add_argument("--inf_num_beams", type=int, default=1)
    
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module("models." + args.model)

    if args.load_best_args:
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=DATASET_NAMES,
            help="Which dataset to perform experiments on.",
        )
        if hasattr(mod, "Buffer"):
            parser.add_argument(
                "--buffer_size",
                type=int,
                required=True,
                help="The size of the memory buffer.",
            )
        args = parser.parse_args()
        if args.model == "joint":
            best = best_args[args.dataset]["sgd"]
        else:
            best = best_args[args.dataset][args.model]
        if args.model == "joint" and args.dataset == "mnist-360":
            args.model = "joint_gcl"
        if hasattr(args, "buffer_size"):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, "get_parser")
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    setattr(args, "GAN", "GAN")
    setattr(args, "use_albumentations", False)
    setattr(args, "use_apex", False)
    setattr(args, "use_distributed", True)
    if "imagenet" in args.dataset:
        setattr(args, "use_lr_scheduler", True)
    else:
        setattr(args, "use_lr_scheduler", False)
    if torch.cuda.device_count() <= 1 or args.dataset == "seq-mnist":
        setattr(args, "use_distributed", False)

    if args.model == "mer":
        setattr(args, "batch_size", 1)
    dataset = get_dataset(args)
    if args.model == "our" or args.model == "our_reservoir":
        backbone = dataset.get_backbone_our()
    elif args.model == "onlinevt":
        if args.with_brain:
            backbone = dataset.get_backbone_brain_coworker()
        elif args.with_brain_vit:
            backbone = dataset.get_backbone_brain_coworker_vit(args)
        elif args.vit_finetune:
            backbone = dataset.get_backbone_vit_pre(args)
        else:
            backbone = dataset.get_backbone_cct()
    else:
        backbone = dataset.get_backbone()
    print('Backbone loaded done')
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    #print(model)
    if args.use_apex or args.use_distributed:
        dist.init_process_group(backend="nccl")  # , init_method='env://'
        torch.cuda.set_device(args.local_rank)
        model.to(model.device)

        if args.use_apex:
            model = convert_syncbn_model(model)
            model.net, model.opt = amp.initialize(model.net, model.opt, opt_level="O1")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        print("Let's use", torch.cuda.device_count(), "GPUs!!!")
        if hasattr(model.net, "net") and hasattr(model.net.net, "distill_classifier"):
            distill_classifier = model.net.net.distill_classifier
            distill_classification = model.net.net.distill_classification
            update_gamma = model.net.net.update_gamma
            if args.use_apex:
                print("Let's use apex !!!")
                model.net.net = DDP_apex(
                    model.net.net, delay_allreduce=True
                )  # , device_ids=[args.local_rank]
            else:
                model.net.net = DDP(
                    model.net.net,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            setattr(model.net.net, "distill_classifier", distill_classifier)
            setattr(model.net.net, "distill_classification", distill_classification)
            setattr(model.net.net, "update_gamma", update_gamma)
        else:
            get_params = model.net.get_params
            model.net = DDP(
                model.net,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )  # , device_ids=[args.local_rank]
            setattr(model.net, "get_params", get_params)
            pass
    else:
        model.to(model.device)

    print(args)
    if hasattr(model, "loss_name"):
        print("loss name:  ", model.loss_name)

    if isinstance(dataset, ContinualDataset):
        if args.diffumem:
            train_diffusion(model, dataset, args)
        else:
            train(model, dataset, args)
    else:
        assert not hasattr(model, "end_task") or model.NAME == "joint_gcl"
        ctrain(args)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(42)
    # setup_seed(2345)
    # setup_seed(1234)
    # setup_seed(3456)
    main()
