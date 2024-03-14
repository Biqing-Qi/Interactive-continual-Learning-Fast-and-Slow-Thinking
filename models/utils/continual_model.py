import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch import optim
from .scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """

    NAME = None
    COMPATIBILITY = []

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        args: Namespace,
        transform: torchvision.transforms,
    ) -> None:
        super(ContinualModel, self).__init__()
        
        self.net = backbone
        self.loss = loss
        self.args = args
        #print(self.args.with_brain_vit)
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        # self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.wd_reg)
        # self.opt = optim.Adam(self.net.parameters())  # weight_decay=self.args.wd_reg, lr=self.args.lr
        if "imagenet" in self.args.dataset:
            self.opt_f = Adam(
                    self.net.parameters(), lr=1e-4, weight_decay=0.000
                )
        elif 'cifar' in self.args.dataset:
            self.opt_f = Adam(
                self.net.parameters(), lr=0.0001, weight_decay=0.000
            )
        self._scheduler = None
        #if self.args.use_lr_scheduler:
        self.set_opt()

        self.device = get_device(args)

    def check_name(self, name_check, namelist):
            out = False
            for name in namelist:
                if name in name_check:
                    out = True
                    break
            return out

    def set_opt(self):
        if "imagenet" in self.args.dataset:
            print(self.args.dataset, " using lr_scheduler.MultiStepLR !!")
            #weight_decay = 0.0005
            weight_decay = 0
            _scheduling = [30, 60, 80, 90]
            lr_decay = 0.999
            if self.args.with_brain:
                brain_params = list(self.net.net.Brain_embedding.parameters()) + list(self.net.net.embedding_map.parameters())
                other_params = list(self.net.net.conv.parameters()) + list(self.net.net.pool.parameters()) + list(self.net.net.mlp_head.parameters()) + list(self.net.net.memory_map.parameters()) + list(self.net.net.history_map.parameters())
                self.opt_mem = SGD(
                    brain_params, lr=0.03, weight_decay=weight_decay
                )
                self.opt_other = SGD(
                    other_params, lr=0.1, weight_decay=weight_decay
                )
            elif self.args.with_brain_vit:
                brain_params = list(self.net.net.memory_2btrain.parameters()) + list(self.net.net.memory_2btrain_tsk.parameters())
                proj_params = list(self.net.net.memory_map.parameters()) + list(self.net.net.embedding_map.parameters()) #+ list(self.net.net.external_att.parameters())
                in_params = []
                if self.net.net.vit is not None:
                    self.memory_names = ['memory_2btrain', 'memory_2btrain_tsk']
                    self.memory_in = ['memoryin_2btrain']
                    self.proj_names = ['memory_map', 'embedding_map']
                    for i, layer in enumerate(self.net.net.vit.model.vit.encoder.layer):
                        for name, module in layer.named_modules():
                            if 'attention' in name:
                                for param_name, param in module.named_parameters():
                                    # if 'memory_params' in param_name:
                                    #     brain_params.append(param)
                                    if self.check_name(param_name, self.memory_names):
                                        brain_params.append(param)
                                        
                                    elif self.check_name(param_name, self.proj_names):
                                        proj_params.append(param)
                                        print(param_name)
                                    
                                    elif self.check_name(param_name, self.memory_in):
                                        in_params.append(param)
                else:
                    proj_params.extend(list(self.net.net.external_att.parameters()))
                    #print(list(self.net.net.external_att.parameters()))
                self.opt_brain = Adam(
                    brain_params, lr=1e-4, weight_decay=weight_decay
                )
                self.opt_proj = Adam(
                    proj_params, lr=1e-4, weight_decay=weight_decay
                )


        elif 'cifar' in self.args.dataset and self.args.with_brain_vit:
            weight_decay = 0.000
            _scheduling = [30, 60, 80, 90]
            lr_decay = 0.7
            brain_params = list(self.net.net.memory_2btrain.parameters()) + list(self.net.net.memory_2btrain_tsk.parameters())
            proj_params = list(self.net.net.memory_map.parameters()) + list(self.net.net.embedding_map.parameters()) #+ list(self.net.net.external_att.parameters())
            in_params = []
            if self.net.net.vit is not None:
                self.memory_names = ['memory_2btrain', 'memory_2btrain_tsk']
                self.memory_in = ['memoryin_2btrain']
                self.proj_names = ['memory_map', 'embedding_map']
                for i, layer in enumerate(self.net.net.vit.model.vit.encoder.layer):
                    for name, module in layer.named_modules():
                        if 'attention' in name:
                            for param_name, param in module.named_parameters():
                                # if 'memory_params' in param_name:
                                #     brain_params.append(param)
                                if self.check_name(param_name, self.memory_names):
                                    brain_params.append(param)
                                    
                                elif self.check_name(param_name, self.proj_names):
                                    proj_params.append(param)
                                    print(param_name)
                                
                                elif self.check_name(param_name, self.memory_in):
                                    in_params.append(param)
            else:
                proj_params.extend(list(self.net.net.external_att.parameters()))
                #print(list(self.net.net.external_att.parameters()))
            self.opt_brain = Adam(
                brain_params, lr=0.0001, weight_decay=weight_decay
            )
            self.opt_proj = Adam(
                proj_params, lr=0.0001, weight_decay=weight_decay
            )
            # self.opt_mem = Adam(
            #     in_params, lr=1e-3, weight_decay=weight_decay
            # )
        else:
            weight_decay = 0
            _scheduling = [30, 60, 80, 90]
            lr_decay = 0.1
            # 获取所有需要优化的参数
            #params = [p for p in self.net.parameters() if p.requires_grad]

            # 创建优化器
            #self.opt = SGD(params, lr=self.args.lr, weight_decay=weight_decay)
            self.opt = SGD(
                self.net.parameters(), lr=self.args.lr, weight_decay=weight_decay
            )
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, _scheduling, gamma=lr_decay
            )
        if self.args.vit_finetune:
            pass
        else:
            self.scheduler_mem = torch.optim.lr_scheduler.MultiStepLR(
                self.opt_brain, _scheduling, gamma=lr_decay
                )
            self.scheduler_proj = torch.optim.lr_scheduler.MultiStepLR(
                self.opt_proj, _scheduling, gamma=lr_decay
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        # print("\tIn Model: input size", x.size())
        return self.net(x)

    def observe(
        self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor
    ) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
