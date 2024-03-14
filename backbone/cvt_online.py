from math import ceil
import torch
from torchvision import models
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.functional import relu
from kornia.augmentation import (
    Resize,
    Normalize
)
# helpers
from torch.nn.utils import spectral_norm
from torch.nn import init
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from label2prompt import label2prompt
from transformers import CLIPModel, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTAttention
import timm
import torch.nn.init as init

class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )
        if self.sigma is not None:
            out = self.sigma * out
        return out


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, l=3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))


def always(val):
    return lambda *args, **kwargs: val


# ResNet18
class Bottleneck(nn.Module):
    expansion = 4  # # output cahnnels / # input channels

    def __init__(self, inplanes, outplanes, stride=1):
        assert outplanes % self.expansion == 0
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.bottleneck_planes = int(outplanes / self.expansion)
        self.stride = stride

        self._make_layer()

    def _make_layer(self):
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(
            self.inplanes,
            self.bottleneck_planes,
            kernel_size=1,
            stride=self.stride,
            bias=False,
        )
        # conv 3x3
        self.bn2 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv2 = nn.Conv2d(
            self.bottleneck_planes,
            self.bottleneck_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # conv 1x1
        self.bn3 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv3 = nn.Conv2d(
            self.bottleneck_planes, self.outplanes, kernel_size=1, stride=1
        )
        if self.inplanes != self.outplanes:
            self.shortcut = nn.Conv2d(
                self.inplanes,
                self.outplanes,
                kernel_size=1,
                stride=self.stride,
                bias=False,
            )
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out


class ResNet164(nn.Module):
    def __init__(self):
        super(ResNet164, self).__init__()
        nstages = [16, 64, 128, 256]
        # one conv at the beginning (spatial size: 32x32)
        self.conv1 = nn.Conv2d(
            3, nstages[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        depth = 164
        block = Bottleneck
        n = int((depth - 2) / 9)
        # use `block` as unit to construct res-net
        # Stage 0 (spatial size: 32x32)
        self.layer1 = self._make_layer(block, nstages[0], nstages[1], n)
        # Stage 1 (spatial size: 32x32)
        self.layer2 = self._make_layer(block, nstages[1], nstages[2], n, stride=2)
        # Stage 2 (spatial size: 16x16)
        self.layer3 = self._make_layer(block, nstages[2], nstages[3], n, stride=2)
        # Stage 3 (spatial size: 8x8)
        self.bn = nn.BatchNorm2d(nstages[3])
        self.relu = nn.ReLU(inplace=True)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, nstage, stride=1):
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        for i in range(1, nstage):
            layers.append(block(outplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


####################################################################################################


# ResNet18
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out, inplace=True)
        return out


class ResNet18Pre(nn.Module):
    def __init__(self, nf=32, stages=3):
        super(ResNet18Pre, self).__init__()
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2)
        self._resnet_high = nn.Sequential(self.layer4, nn.Identity())
        if nf == 64:
            if self.stages == 3:
                self.resnet_low = nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.relu,
                    self.layer1,  # 64, 32, 32
                    self.layer2,  # 128, 16, 16
                    # self.layer3,  # 256, 8, 8
                )
            if self.stages == 2:
                self.resnet_low = nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.relu,
                    self.layer1,  # 64, 32, 32
                    self.layer2,  # 128, 16, 16
                    self.layer3,  # 256, 8, 8
                )

        else:
            self.resnet_low = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.layer1,  # nf, h, w
                self.layer2,  # 2*nf, h/2, w/2
                self.layer3,  # 4*nf, h/4, w/4
                self.layer4,  # 8*nf, h/8, w/8
            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=128, width=128):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(
        self, block: BasicBlock, planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class ResNet18Pre128(nn.Module):
    def __init__(self, stages):
        super(ResNet18Pre128, self).__init__()

        nf = 64
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        ) # kernel_size=3, stride=1, padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2) # layer4
        self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2) 
        # self._resnet_high = nn.Sequential(
        #                                   self.layer4,
        #                                   nn.Identity()
        #                                   )
        if self.stages == 2:
            self.resnet_low = nn.Sequential(
                self.conv1,
                self.maxpool,
                self.bn1,
                self.relu,
                self.layer1,  # 64, 32, 32
                self.layer2,  # 128, 16, 16
                self.layer3,  # 256, 8, 8
                self.layer4
            )
        if self.stages == 3:
            self.resnet_low = nn.Sequential(
                self.conv1,
                self.maxpool,
                self.bn1,
                self.relu,
                self.layer1,  # 64, 32, 32
                self.layer2,  # 128, 16, 16
                self.layer3,  # 256, 8, 8
                self.layer4
            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=128, width=128):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(
        self, block: BasicBlock, planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout=0.0, SN=False):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, dim * mult, 1))
            if SN
            else nn.Conv2d(dim, dim * mult, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv2d(dim * mult, dim, 1))
            if SN
            else nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(
            x, dim=1, unbiased=False, keepdim=True
        ).sqrt()  # dim=1 channel-wise
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ExternalAttention_module(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        heads=8,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        BN=True,
        SN=False,
    ):
        super().__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key**-0.5

        if SN:
            self.to_q = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        dim,
                        inner_dim_key,
                        1,
                        stride=(2 if downsample else 1),
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_k = nn.Sequential(
                spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, bias=False)),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias=False)),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )
        else:
            self.to_q = nn.Sequential(
                nn.Conv2d(
                    dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_k = nn.Sequential(
                nn.Conv2d(dim, inner_dim_key, 1, bias=False),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                nn.Conv2d(dim, inner_dim_value, 1, bias=False),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )

        self.attend = nn.Softmax(dim=-1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1))
            if SN
            else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout),
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step=(2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, "i j c -> (i j) c"), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim=-1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer("pos_indices", pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, "i j h -> () h i j")
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, "b (h d) ... -> b h (...) d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=h, y=y)
        return self.to_out(out)


class AttentionDIY(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        heads=8,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        BN=True,
        SN=False,
    ):
        super().__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key**-0.5

        if SN:
            self.to_q = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        dim,
                        inner_dim_key,
                        1,
                        stride=(2 if downsample else 1),
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias=False)),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )
        else:
            self.to_q = nn.Sequential(
                nn.Conv2d(
                    dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                nn.Conv2d(dim, inner_dim_value, 1, bias=False),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )

        self.attend = nn.Softmax(dim=-1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        # self.external_k = nn.Sequential(
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
        #     out_batch_norm if BN else nn.Identity(),
        #     nn.Dropout(dropout)
        # )

        self.mk_batch_norm = nn.BatchNorm2d(fmap_size * fmap_size)
        self.mk = nn.Sequential(
            nn.Linear(dim_key, fmap_size * fmap_size, bias=False),
            # mk_batch_norm
        )

        self.to_out = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1))
            if SN
            else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout),
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step=(2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, "i j c -> (i j) c"), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim=-1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer("pos_indices", pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, "i j h -> () h i j")
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qv = (q, self.to_v(x))

        q, v = map(lambda t: rearrange(t, "b (h d) ... -> b h (...) d", h=h), qv)
        # q = map(lambda t: rearrange(t, 'b (h d) ... -> b h d (...)', h=h), q)

        dots = self.mk(q)

        dots = rearrange(dots, "b h hw d -> b d hw h")

        dots = self.mk_batch_norm(dots)

        dots = rearrange(dots, "b d hw h -> b h hw d")

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S  效果不好，而且波动加大

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=h, y=y)
        return self.to_out(out)


class AttentionDIYbn(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        heads=8,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        BN=True,
        SN=False,
    ):
        super().__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key**-0.5

        if SN:
            self.to_q = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        dim,
                        inner_dim_key,
                        1,
                        stride=(2 if downsample else 1),
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias=False)),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )
        else:
            self.to_q = nn.Sequential(
                nn.Conv2d(
                    dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False
                ),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity(),
            )
            self.to_v = nn.Sequential(
                nn.Conv2d(dim, inner_dim_value, 1, bias=False),
                nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity(),
            )

        self.mk = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    inner_dim_key, self.heads * fmap_size * fmap_size, 1, bias=False
                ),
                nn.BatchNorm2d(self.heads * fmap_size * fmap_size),
            ),
        )

        self.attend = nn.Softmax(dim=-1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        # self.external_k = nn.Sequential(
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
        #     out_batch_norm if BN else nn.Identity(),
        #     nn.Dropout(dropout)
        # )

        self.to_out = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1))
            if SN
            else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout),
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step=(2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, "i j c -> (i j) c"), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim=-1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer("pos_indices", pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, "i j h -> () h i j")
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        v = rearrange(self.to_v(x), "b (h d) ... -> b h (...) d", h=h)

        dots = self.mk(q)

        dots = rearrange(dots, "b (h d) ... -> b h (...) d", h=h)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S  效果不好，而且波动加大

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=h, y=y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        depth,
        heads,
        dim_key,
        dim_value,
        mlp_mult=2,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        BN=True,
        SN=False,
        LN=False,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        if LN:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                AttentionDIYbn(
                                    dim,
                                    fmap_size=fmap_size,
                                    heads=heads,
                                    dim_key=dim_key,
                                    dim_value=dim_value,
                                    dropout=dropout,
                                    downsample=downsample,
                                    dim_out=dim_out,
                                    BN=BN,
                                    SN=SN,
                                ),
                            ),
                            PreNorm(
                                dim_out,
                                FeedForward(dim_out, mlp_mult, dropout=dropout, SN=SN),
                            ),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            AttentionDIYbn(
                                dim,
                                fmap_size=fmap_size,
                                heads=heads,
                                dim_key=dim_key,
                                dim_value=dim_value,
                                dropout=dropout,
                                downsample=downsample,
                                dim_out=dim_out,
                                BN=BN,
                                SN=SN,
                            ),
                            FeedForward(dim_out, mlp_mult, dropout=dropout, SN=SN),
                        ]
                    )
                )

    def forward(self, x):
        # x (10, 128, 7, 7)
        for attn, ff in self.layers:
            attn_res = x if self.attn_residual else 0
            x = attn(x) + attn_res
            x = ff(x) + x
        return x


class CVT_online(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_mult,
        stages=3,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
        cnnbackbone="ResNet18Pre",
        independent_classifier=False,
        frozen_head=False,
        BN=True,  # Batchnorm
        LN=False,  # LayerNorm
        SN=False,  # SpectralNorm
        grow=False,  # Expand the network
        mean_cob=True,
        sum_cob=False,
        max_cob=False,
        distill_classifier=True,
        cosine_classifier=False,
        use_transformer = False,
        use_WA=False,
        init="kaiming",
        device="cuda",
        use_bias=True,
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        self.dims = dims
        self.depths = depths
        self.layer_heads = layer_heads
        self.image_size = image_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_mult = mlp_mult
        self.stages = stages
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout = dropout
        self.distill_classifier = distill_classifier
        self.cnnbackbone = cnnbackbone
        if image_size == 128:
            self.cnnbackbone = "ResNet18Pre128"  # ResNet18Pre224   PreActResNet
        self.nf = 64 if image_size < 100 else 64
        self.independent_classifier = independent_classifier
        self.frozen_head = frozen_head
        self.BN = BN
        self.SN = SN
        self.LN = LN
        self.grow = grow
        self.init = init
        self.use_WA = use_WA
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_transformer = use_transformer
        self.use_bias = use_bias
        self.mean_cob = mean_cob
        self.sum_cob = sum_cob
        self.max_cob = max_cob
        self.gamma = None

        print("-----------------------------------------", depths)
        assert all(
            map(lambda t: len(t) == stages, (dims, depths, layer_heads))
        ), "dimensions, depths, and heads must be a tuple that is less than the designated number of stages"

        if self.cnnbackbone == "ResNet18Pre":
            self.conv = ResNet18Pre(self.nf, self.stages)
        elif self.cnnbackbone == "ResNet164":
            self.conv = ResNet164()
        elif self.cnnbackbone == "ResNet18Pre128":
            print("Backbone: ResNet18Pre128")
            self.conv = ResNet18Pre128(self.stages)
            # self.conv = models.resnet18(pretrained=True)
        elif self.cnnbackbone == "PreActResNet":
            self.conv = PreActResNet()
        else:
            assert ()
        if use_transformer:
            if grow:
                print("Enable dynamical Transformer expansion!")
                self.transformers = nn.ModuleList()
                self.transformers.append(self.add_transformer())
                # self.transformers.append(self.add_transformer())
                # self.transformers.append(self.add_transformer())
                # self.transformers.append(self.add_transformer())
                # self.transformers.append(self.add_transformer())
            else:
                self.transformer = (
                    self.add_transformer()
                )  # self.add_transformer()  # self.conv._resnet_high

        self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), Rearrange("... () () -> ...")
            )


        self.distill_head = (
            self._gen_classifier(dims[-1], num_classes)
            if self.distill_classifier
            else always(None)
        )

        if self.independent_classifier:
            task_class = 2 if num_classes < 20 else 20
            self.fix = nn.ModuleList(
                [
                    self._gen_classifier(dims[-1], task_class)
                    for i in range(num_classes // task_class)
                ]
            )
        else:
            self.mlp_head = (
                spectral_norm(self._gen_classifier(dims[-1], num_classes))
                if SN
                else self._gen_classifier(dims[-1], num_classes)
            )

        self.feature_head = self._gen_classifier(dims[-1], num_classes)

        # self.focuses = nn.Parameter(torch.FloatTensor(self.num_classes, self.dims[-1]), requires_grad=True).to(self.device)
        self.focuses = nn.Parameter(
            torch.FloatTensor(self.num_classes, 512).fill_(1), requires_grad=True
        ).to(self.device)
        # self.focuses = F.normalize(focuses_org, dim=1)
        self.focus_labels = torch.tensor([i for i in range(self.num_classes)]).to(
            self.device
        )

    def focuses_head(self):
        return F.normalize(self.feature_head(self.focuses), dim=1)

    def add_transformer(self):
        if self.nf == 64:
            fmap_size = self.image_size // ((2**2) if self.stages < 3 else (2**1))
        else:
            fmap_size = self.image_size // (2**3)
        if self.cnnbackbone == "ResNet18Pre128" or self.cnnbackbone == "PreActResNet":
            if self.stages == 3:
                fmap_size = self.image_size // (2**4)
            if self.stages == 2:
                fmap_size = self.image_size // (2**3)
        layers = []

        for ind, dim, depth, heads in zip(
            range(self.stages), self.dims, self.depths, self.layer_heads
        ): # dim (128, 256, 512)
            is_last = ind == (self.stages - 1)
            layers.append(
                Transformer(
                    dim,
                    fmap_size,
                    depth,
                    heads,
                    self.dim_key,
                    self.dim_value,
                    self.mlp_mult,
                    self.dropout,
                    BN=self.BN,
                    SN=self.SN,
                    LN=self.LN,
                )
            )

            if not is_last:  # downsample
                next_dim = self.dims[ind + 1]
                layers.append(
                    Transformer(
                        dim,
                        fmap_size,
                        1,
                        heads * 2,
                        self.dim_key,
                        self.dim_value,
                        dim_out=next_dim,
                        downsample=True,
                        BN=self.BN,
                        SN=self.SN,
                        LN=self.LN,
                    )
                )
                fmap_size = ceil(fmap_size / 2)
        return nn.Sequential(*layers)

    def fix_and_grow(self):
        print("fix and grow !!!")
        # for param in self.conv.parameters():
        #     param.requires_grad = False
        # # self.conv.eval()
        # for param in self.transformers.parameters():
        #     param.requires_grad = False
        # # self.transformers.eval()
        self.transformers.append(self.add_transformer())
        # return self

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)
        return classifier

    @torch.no_grad()
    def update_gamma(self, task_num, class_per_task):
        if task_num == 0:
            return 1
        if self.distill_classifier:
            classifier = self.distill_head
        else:
            classifier = self.mlp_head
        old_weight_norm = torch.norm(
            classifier.weight[: task_num * class_per_task], p=2, dim=1
        )
        new_weight_norm = torch.norm(
            classifier.weight[
                task_num * class_per_task : task_num * class_per_task + class_per_task
            ],
            p=2,
            dim=1,
        )
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print("gamma: ", self.gamma.cpu().item(), "  use_WA:", self.use_WA)
        if not self.use_WA:
            return 1
        return self.gamma

    def forward(self, img):
        x = self.conv(img)
        if self.use_transformer:
            if self.grow:
                x = [transformer(x) for transformer in self.transformers]
                if self.sum_cob:
                    x = torch.stack(x).sum(dim=0)  # add growing transformers' output
                elif self.mean_cob:
                    x = torch.stack(x).mean(dim=0)
                elif self.max_cob:
                    for i in range(len(x) - 1):
                        x[i + 1] = x[i].max(x[i + 1])
                    x = x[-1]
                else:
                    ValueError
            else:
                x = self.transformer(x)
        x = self.pool(x)

        if self.independent_classifier:
            y = torch.tensor([])
            for fix in self.fix:
                y = torch.cat((fix(x), y), 1)
            out = y
        else:
            out = self.mlp_head(x)

        # print('Out size:', out.size())

        return out

    def distill_classification(self, img):
        # with torch.cuda.amp.autocast():
        x = self.conv(img)
        if self.use_transformer:
            if self.grow:
                x = [transformer(x) for transformer in self.transformers]
                if self.sum_cob:
                    x = torch.stack(x).sum(dim=0)  # add growing transformers' output
                elif self.mean_cob:
                    x = torch.stack(x).mean(dim=0)
                elif self.max_cob:
                    for i in range(len(x) - 1):
                        x[i + 1] = x[i].max(x[i + 1])
                    x = x[-1]
            else:
                x = self.transformer(x)

        x = self.pool(x)
        distill = self.distill_head(x)

        if self.independent_classifier:
            y = torch.tensor([]).to("cuda")
            for fix in self.fix:
                y = torch.cat((fix(x).to("cuda"), y), 1)
            out = y
        else:
            out = self.mlp_head(x)

        if exists(distill):
            return distill
        # print('distill_classification Out size:', out.size())

        return out

    def contrasive_f(self, img):
        x = self.conv(img)
        if self.use_transformer:
            if self.grow:
                x = [transformer(x) for transformer in self.transformers]
                if self.sum_cob:
                    x = torch.stack(x).sum(dim=0)  # add growing transformers' output
                elif self.mean_cob:
                    x = torch.stack(x).mean(dim=0)
                elif self.max_cob:
                    for i in range(len(x) - 1):
                        x[i + 1] = x[i].max(x[i + 1])
                    x = x[-1]
                else:
                    ValueError
            else:
                x = self.transformer(x)

        x = self.pool(x)
        x = self.feature_head(x)  # 去掉效果好像略好，区别不太大！！
        x = F.normalize(x, dim=1)

        return x

    def frozen(self, t):
        if self.independent_classifier and self.frozen_head:
            print("----------frozen-----------")
            for i in range(t + 1):
                self.fix[i].weight.requires_grad = False
                print("frozen ", i)
        if t > -1 and self.grow:
            self.fix_and_grow()
            pass

class ResNet18_Brain(nn.Module):
    def __init__(self):
        super(ResNet18_Brain, self).__init__()

        nf = 64
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        ) # kernel_size=3, stride=1, padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2) # layer4
        self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2) 
        self.resnet = nn.Sequential(
                self.conv1,
                self.maxpool,
                self.bn1,
                self.relu,
                self.layer1,  # 64, 32, 32
                self.layer2,  # 128, 16, 16
                self.layer3,  # 256, 8, 8
                self.layer4
            )

        self.apply(self.init_weight)

    def _make_layer(
        self, block: BasicBlock, planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class Brain_Vit(nn.Module):
    def __init__(self):
        super(Brain_Vit, self).__init__()
        #self.model = CLIPModel.from_pretrained("/home/bqqi/.cache/huggingface/transformers/clip-vit-large-patch14")
        #self.model = CLIPModel.from_pretrained("/home/bqqi/.cache/huggingface/transformers/clip-vit-base-patch16")
        
        #model_path = '/home/bqqi/lifelong_research/base_model'
        #model = timm.create_model("vit_base_patch16_224", pretrained=False, checkpoint_path=model_path)
        #self.model = CLIPModel.from_pretrained("/home/bqqi/.cache/huggingface/transformers/CLIP-ViT-B-16-laion2B-s34B-b88K")
        self.model = ViTForImageClassification.from_pretrained('/home/bqqi/.cache/huggingface/transformers/vit-base-patch16-224')

        print(self.model)
    def forward(self, x=None, text=None):
        if text is not None:
            output = self.model.text_model.encoder(text).last_hidden_state[:,0,:]
        else:
            with torch.no_grad():
                #output = self.model.get_image_features(x)
                #output = self.model.vision_model(x).last_hidden_state.mean(dim=1)
                #output = self.model.visual_projection(output).mean(dim=1)
                #print(output.shape)
                output = self.model.vit(x).last_hidden_state.mean(dim=1)
            #print(output.shape)
        return output

class Brain_Vit_att(nn.Module):
    def __init__(self):
        super(Brain_Vit_att, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('/home/bqqi/.cache/huggingface/transformers/vit-base-patch16-224').eval()
        # self.model = ViTForImageClassification.from_pretrained('/home/bqqi/.cache/huggingface/transformers/vit-base-patch16-224-in21k')
        self.model.requires_grad = False
        # self.model = CLIPModel.from_pretrained("/home/bqqi/.cache/huggingface/transformers/clip-vit-large-patch14")
        # print(self.model)
    def forward(self, x=None, text=None):
        if text is not None:
            output = self.model.text_model.encoder(text).last_hidden_state[:,0,:]
        else:
            # with torch.no_grad():

            output = self.model.vit(x).last_hidden_state
                # output = self.model.vision_model(x).last_hidden_state
                # output = self.model.visual_projection(output)
        return output

class CustomViTAttention(ViTAttention):
    def __init__(self, num_classes, task_num, config):
        super().__init__(config)
        self.num_classes = num_classes
        self.task_num = task_num
        self.cls_per_tsk = int(num_classes/task_num)

        # self.Brain_embedding = nn.Embedding(num_classes, 576)
        # nn.init.normal_(self.Brain_embedding.weight, mean=0, std=1)

        # self.memory_2btrain = nn.Embedding(num_classes, 576)
        # nn.init.normal_(self.memory_2btrain.weight, mean=0, std=1)

        self.Brain_embedding_tsk = nn.Embedding(task_num, 768)
        nn.init.normal_(self.Brain_embedding_tsk.weight, mean=0, std=1)

        self.memory_2btrain_tsk = nn.Embedding(task_num, 768)
        nn.init.normal_(self.memory_2btrain_tsk.weight, mean=0, std=1)

        # self.memoryin = nn.Embedding(task_num, 197)
        # nn.init.normal_(self.memoryin.weight, mean=0, std=1)

        # self.memoryin_2btrain = nn.Embedding(task_num, 197)
        # nn.init.normal_(self.memoryin_2btrain.weight, mean=0, std=1)

        # self.memory_map = self._gen_projector(768, 3072, 576)
        #self.embedding_map = self._gen_projector(768, 3072, 192)
        self.opened_memories = []
        self.opened_tasks = []

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gen_projector(self, in_features, hidden_dim, out_dim):
        #projector = nn.Linear(in_features, out_dim, bias = False)
        projector = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        #projector = ResidualLinear(in_features, hidden_dim, out_dim)
        projector.apply(self.initialize_weights)
        return projector

    def check_memories(self, labels):  
        for label in labels.unique():
            if label.item() not in self.opened_memories:
                self.opened_memories.append(label.item())
        self.opened_memories.sort()

    def check_tasks(self, tasks):  
        for task in tasks.unique():
            if task.item() not in self.opened_tasks:
                self.opened_tasks.append(task.item())
        self.opened_tasks.sort()

    def label_task(self, labels):
        tasks = torch.zeros_like(labels)
        bin_num = self.num_classes//self.cls_per_tsk
        for i in range(bin_num):
            bin_start = i * self.cls_per_tsk
            bin_end = (i+1) * self.cls_per_tsk
            tasks[(labels >= bin_start) & (labels < bin_end)] = i
        return tasks

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
        tasks: torch.Tensor = None,
        task_now: int = None,
    ):
        if task_now is not None:
            # self.check_memories(labels)
            # condition = labels >= task_now * self.cls_per_tsk
            # brain_embeddings_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels), self.Brain_embedding(labels))
            #embeddings_2bin = torch.where(condition.unsqueeze(1), self.memoryin_2btrain(labels), self.memoryin(labels))
            #tsk = self.label_task(labels)
            self.check_tasks(tasks)
            condition = tasks >= task_now
            brain_embeddings_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(tasks), self.Brain_embedding_tsk(tasks))
            #brain_embeddings = torch.cat([brain_embeddings_tsk, brain_embeddings_cls], dim=-1)
            # brain_embeddings = brain_embeddings_tsk
            # embeddings_2bin = torch.where(condition.unsqueeze(1), self.memoryin_2btrain(tsk), self.memoryin(tsk))
            # query_feature = hidden_states.mean(dim=1)
            # query_feature_cls = self.memory_map(query_feature)
            # query_featrue_tsk = self.embedding_map(query_feature)
            # query_feature = torch.cat([query_featrue_tsk, query_feature_cls], dim=-1)
            #query_feature = query_featrue_tsk
            # query_feature = F.normalize(query_feature, dim=-1)
            # brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            # logit_brain_mem = torch.matmul(query_feature, brain_embeddings.t())
            # hidden_states = brain_embeddings_tsk.unsqueeze(-1) + hidden_states
            #hidden_states = torch.cat([brain_embeddings_tsk.unsqueeze(1), hidden_states], dim=1)
            outputs = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            #outputs = brain_embeddings_tsk.unsqueeze(-1) + outputs[0]
            outputs = torch.cat([brain_embeddings_tsk.unsqueeze(1), outputs[0]], dim=1)
            return (outputs[0],)#, logit_brain_mem
        
        else:
            #query_feature = hidden_states.mean(dim=1)
            # query_feature_cls = self.memory_map(query_feature)
            # query_featrue_tsk = self.embedding_map(query_feature)
            # query_feature = torch.cat([query_featrue_tsk, query_feature_cls], dim=-1)
            #query_feature = query_featrue_tsk
            # brain_embeddings_cls = self.Brain_embedding(torch.tensor(self.opened_memories, device=query_feature.device))
            # brain_embeddings_tsk = self.Brain_embedding_tsk(torch.tensor(self.opened_tasks, device=query_feature.device))
            brain_embeddings_tsk = self.Brain_embedding_tsk(tasks)
            # brain_embeddings = torch.cat([brain_embeddings_tsk.repeat_interleave(self.cls_per_tsk, dim=0), brain_embeddings_cls], dim=-1)
            # brain_embeddings = brain_embeddings_tsk
            # embeddings_2bin = self.memoryin(torch.tensor(self.opened_memories, device=query_feature.device))
            #embeddings_2bin = self.memoryin(torch.tensor(self.opened_tasks, device=query_feature.device))
            # normalized_brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            # normalized_query_feature = F.normalize(query_feature, dim=-1)
            # logit_brain_mem = torch.matmul(normalized_query_feature, normalized_brain_embeddings.t())
            # max_indices = torch.argmax(logit_brain_mem, dim=1)
            # selected_embeddings = torch.index_select(embeddings_2bin, dim=0, index=max_indices)
            #hidden_states = brain_embeddings_tsk.unsqueeze(-1) + hidden_states
            outputs = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            #outputs = brain_embeddings_tsk.unsqueeze(-1) + outputs[0]
            outputs = torch.cat([brain_embeddings_tsk.unsqueeze(1), outputs[0]], dim=1)
            #outputs = torch.cat([outputs[0], selected_embeddings.unsqueeze(1)], dim=1)
            return (outputs[0],)
    
class Brain_Vit_em(nn.Module):
    def __init__(self, num_classes, task_num):
        super(Brain_Vit_em, self).__init__()
        #self.model = CLIPModel.from_pretrained("/home/bqqi/.cache/huggingface/transformers/clip-vit-large-patch14")
        self.model = ViTForImageClassification.from_pretrained('/home/bqqi/.cache/huggingface/transformers/vit-base-patch16-224')
        self.memory_layers = [3, 7, 9]
        self.memory_names = ['Brain_embedding', 'memory_2btrain', 'Brain_embedding_tsk', 'memory_2btrain_tsk']
        self.proj_names = ['memory_map', 'embedding_map', 'memoryin', 'memoryin_2btrain']
        config = self.model.config
        for i, layer in enumerate(self.model.vit.encoder.layer):
            if i in self.memory_layers:
                layer.attention = CustomViTAttention(num_classes, task_num, config)
                for name, module in layer.named_modules():
                    if isinstance(module, CustomViTAttention):
                        for param_name, param in module.named_parameters():
                            
                            # if 'memory_params' not in param_name:
                            #     param.requires_grad = False
                            if not self.check_name(param_name, self.memory_names) and not self.check_name(param_name, self.proj_names):
                                param.requires_grad = False
                            
    def check_name(self, name_check, namelist):
        out = False
        for name in namelist:
            if name in name_check:
                out = True
                break
        return out

    def forward(self, x, tsks = None, task_now=None):
        hidden_states = self.model.vit.embeddings(x)
        #logits = []
        if task_now is not None:
            for i, layer in enumerate(self.model.vit.encoder.layer):
                if i in self.memory_layers:
                    self_attention_outputs = layer.attention(
                        layer.layernorm_before(hidden_states), tasks=tsks, task_now=task_now
                    )
                    #logits.append(logit_brain_mem)
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:] 
                    #hidden_states = attention_output + hidden_states
                    hidden_states = attention_output + torch.cat([torch.zeros([hidden_states.shape[0], 1, hidden_states.shape[-1]], device=hidden_states.device), hidden_states], dim=1)
                    layer_output = layer.layernorm_after(hidden_states)
                    layer_output = layer.intermediate(layer_output)

                    layer_output = layer.output(layer_output, hidden_states)

                    outputs = (layer_output,) + outputs
                    hidden_states = outputs[0]
                else:
                    self_attention_outputs = layer.attention(
                        layer.layernorm_before(hidden_states),  
                    )
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:] 

                    hidden_states = attention_output + hidden_states
                    #print(attention_output.shape)
                    #hidden_states = attention_output + torch.cat([hidden_states, torch.zeros([hidden_states.shape[0], 1, hidden_states.shape[-1]], device=hidden_states.device)], dim=1)
                    layer_output = layer.layernorm_after(hidden_states)
                    layer_output = layer.intermediate(layer_output)

                    layer_output = layer.output(layer_output, hidden_states)

                    outputs = (layer_output,) + outputs
                    hidden_states = outputs[0]
            outputs = self.model.vit.layernorm(hidden_states)
            output = outputs.mean(dim=1)
            #print(output.shape)
            return output #, logits
        
        else:
            for i, layer in enumerate(self.model.vit.encoder.layer):
                if i in self.memory_layers:
                    self_attention_outputs = layer.attention(
                            layer.layernorm_before(hidden_states), tasks=tsks
                    )
                    attention_output = self_attention_outputs[0]
                    hidden_states = attention_output + torch.cat([torch.zeros([hidden_states.shape[0], 1, hidden_states.shape[-1]], device=hidden_states.device), hidden_states], dim=1)
                else:
                    self_attention_outputs = layer.attention(
                            layer.layernorm_before(hidden_states)
                    )
                    attention_output = self_attention_outputs[0]
                    hidden_states = attention_output + hidden_states
                outputs = self_attention_outputs[1:] 
                # if i in self.memory_layers:
                #     hidden_states = attention_output + torch.cat([hidden_states, torch.zeros([hidden_states.shape[0], 1, hidden_states.shape[-1]], device=hidden_states.device)], dim=1)
                # else:

                layer_output = layer.layernorm_after(hidden_states)
                layer_output = layer.intermediate(layer_output)

                layer_output = layer.output(layer_output, hidden_states)

                outputs = (layer_output,) + outputs
                hidden_states = outputs[0]
            outputs = self.model.vit.layernorm(hidden_states)
            output = outputs.mean(dim=1)
            return output

#NO Outer Attention
'''class Brain_Net_Vit(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        hidden_dim,
        cls_per_tsk,
        dropout=0.0,
        frozen_head=False,
        SN=False,  # SpectralNorm
        cosine_classifier=False,
        use_normalize = True,
        init="kaiming",
        device="cuda",
        use_bias=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        if image_size != 224:
            self.transform = nn.Sequential(
            Resize(size=(224, 224)),
            Normalize(torch.FloatTensor((0.5, 0.5, 0.5)), torch.FloatTensor((0.5, 0.5, 0.5)))
        )
        else:
            self.transform = False
        self.num_classes = num_classes
        self.use_normalize = use_normalize
        self.dropout = dropout
        self.frozen_head = frozen_head
        self.SN = SN
        self.init = init
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_bias = use_bias
        self.LN = nn.LayerNorm(512)
        
        #self.vit.parameters().requires_grad = False
        #for param in self.vit.parameters():
        #    param.requires_grad = False
        self.opened_memories = []
        self.opened_tasks = []
        self.memory_map = self._gen_projector(768, 3072, 768)
        #self.history_map = self._gen_projector(512, hidden_dim)
        self.task_num = int(num_classes/cls_per_tsk)
        self.query = Brain_Vit()
        self.vit = Brain_Vit_em(num_classes, self.task_num)
        # self.vit =  Brain_Vit()
        # 归一化向量
        #F.normalize(vectors, dim=-1)
        #print(torch.matmul(normalized_vectors, normalized_vectors.t()))

        # Create the embedding layer with the orthogonal matrix

        self.Brain_embedding = nn.Embedding(num_classes, 768)
        nn.init.normal_(self.Brain_embedding.weight, mean=0, std=1)

        self.memory_2btrain = nn.Embedding(num_classes, 768)
        nn.init.normal_(self.memory_2btrain.weight, mean=0, std=1)

        self.Brain_embedding_tsk = nn.Embedding(self.task_num, 192)
        nn.init.normal_(self.Brain_embedding_tsk.weight, mean=0, std=1)

        self.memory_2btrain_tsk = nn.Embedding(self.task_num, 192)
        nn.init.normal_(self.memory_2btrain_tsk.weight, mean=0, std=1)
        #print(torch.eye(768)[num_classes, :].shape)
        #self.memory_2btrain_tsk = nn.Embedding.from_pretrained(torch.eye(768)[:num_classes, :])

        self.task_now = 0
        self.cls_per_tsk = cls_per_tsk
        #self.Brain_embedding = nn.Embedding(num_classes, 512)
        #self.memory_2btrain = nn.Embedding(num_classes, 512)
        self.Brain_embedding.requires_grad = False
        self.embedding_map = self._gen_projector(768, 3072, 192)
        #self.all_map =  nn.Linear(768, 192, bias = False)# self._gen_projector(768, 1024, 512)
        #self.A = nn.Linear(hidden_dim, hidden_dim, bias=False, requires_grad=False)
    
    def label_task(self, labels):
        tasks = torch.zeros_like(labels)
        bin_num = self.num_classes//self.cls_per_tsk
        for i in range(bin_num):
            bin_start = i * self.cls_per_tsk
            bin_end = (i+1) * self.cls_per_tsk
            tasks[(labels >= bin_start) & (labels < bin_end)] = i
        return tasks

    def check_memories(self, labels):  
        for label in labels.unique():
            if label.item() not in self.opened_memories:
                self.opened_memories.append(label.item())
        self.opened_memories.sort()

    def check_tasks(self, tasks):  
        for task in tasks.unique():
            if task.item() not in self.opened_tasks:
                self.opened_tasks.append(task.item())
        self.opened_tasks.sort()

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gen_projector(self, in_features, hidden_dim, out_dim):
        #projector = nn.Linear(in_features, out_dim, bias = False)
        projector = nn.Sequential(nn.Linear(in_features, hidden_dim, bias=self.use_bias), nn.GELU(), nn.Linear(hidden_dim, out_dim, bias=self.use_bias))
        #projector = ResidualLinear(in_features, hidden_dim, out_dim)
        projector.apply(self.initialize_weights)
        return projector
    
    def mask_with_task(self, pred_task, logits):
        mask = torch.zeros_like(logits)
        # if pred_task.dim()>=2:
        #     for i in range(pred_task.shape[0]):
        #         for pred in pred_task[i]:
        #             mask[i][pred.item()*self.cls_per_tsk:(pred.item()+1)*self.cls_per_tsk]=1
        #     masked_logits = mask * logits
        # else:
        for i in range(pred_task.shape[0]):
            mask[i][pred_task[i]*self.cls_per_tsk:(pred_task[i]+1)*self.cls_per_tsk]=1
        masked_logits = mask * logits
        return masked_logits

    def forward(self, img, labels=None):

        if labels is not None:
            self.check_memories(labels)
            if self.transform:
                img = self.transform(img)
            #brain_embeddings = self.Brain_embedding(labels)
            
            condition = labels >= self.task_now * self.cls_per_tsk
            brain_embeddings_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels), self.Brain_embedding(labels))
            tsk = self.label_task(labels)
            self.check_tasks(tsk)
            condition = tsk >= self.task_now
            brain_embeddings_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(tsk), self.Brain_embedding_tsk(tsk))
            #brain_embeddings = torch.cat([brain_embeddings_tsk, brain_embeddings_cls], dim=-1)
            #brain_embeddings = brain_embeddings_cls
            #brain_bias = torch.where(condition.unsqueeze(1), self.bias_2btrain(labels), self.Brain_bias(labels))
            #img_features, logits = (self.vit(img, labels, self.task_now))
            query_features = self.query(img)
            #img_features_cls = self.memory_map(img_features)
            img_features_tsk = self.embedding_map(query_features)
            #img_features = torch.cat([img_features_tsk, img_features_cls], dim=-1)
            img_features_cls = self.memory_map(self.vit(img, tsk, self.task_now))
            #img_features = self.all_map(img_features)
            #img_features = memory_gate + img_features + brain_bias
            #brain_embeddings = self.vit(text=brain_embeddings.view(brain_embeddings.shape[0], -1, 768))
            # labels_all = torch.tensor(self.opened_memories, device=img.device)
            # condition = labels_all >= self.task_now * self.cls_per_tsk
            # all_memories_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels_all), self.Brain_embedding(labels_all))
            # task_all = torch.tensor(self.opened_tasks, device=img.device)
            # condition = task_all >= self.task_now
            # all_memories_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(task_all), self.Brain_embedding_tsk(task_all))
            #print(torch.cat([torch.tensor([self.cls_per_tsk]*self.task_now), torch.tensor([all_memories_cls.shape[0]])], dim=0).int().to(all_memories_tsk.device), all_memories_cls.shape)
            # if all_memories_cls.shape[0] >= self.cls_per_tsk * (self.task_now+1):
            #     all_memories = torch.cat([all_memories_tsk.repeat_interleave(self.cls_per_tsk, dim=0), all_memories_cls], dim=-1)
            # else: 
            #     all_memories = torch.cat([all_memories_tsk.repeat_interleave(torch.cat([torch.tensor([self.cls_per_tsk]*self.task_now), torch.tensor([all_memories_cls.shape[0]%self.cls_per_tsk])], dim=0).int().to(all_memories_tsk.device), dim=0), all_memories_cls], dim=-1)
            #all_memories = all_memories_cls
            #all_memories_mean = self.embedding_map(all_memories).mean(dim=0)
            #img_features = self.embedding_map(img_features) #.mean(dim=1)
            #img_features = self.embedding_map(img_features)
            #img_features = self.memory_map(torch.cat([img_features, all_memories_mean.squeeze(0).repeat(img_features.shape[0],1)], dim=-1))
            #x_brain_mem = self.embedding_map(brain_embeddings)

            if self.use_normalize:
                img_features_cls = F.normalize(img_features_cls, dim=-1)
                brain_embeddings_cls = F.normalize(brain_embeddings_cls, dim=-1)
                img_features_tsk = F.normalize(img_features_tsk, dim=-1)
                brain_embeddings_tsk = F.normalize(brain_embeddings_tsk, dim=-1)
                # img_features = F.normalize(img_features, dim=-1)
                # brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            #y_true = self.A(y_true)
            #y_brain_mem = self.A(y_brain_mem)
            #y_history = self.A(y_history)
            logit_brain_mem = torch.matmul(img_features_cls, brain_embeddings_cls.t())
            logit_embedding_mem = (torch.matmul(img_features_tsk, brain_embeddings_tsk.t()), tsk)
            #print(logit_brain_mem_tsk)
            return logit_brain_mem, None, logit_embedding_mem#, F.normalize(all_memories, dim=-1)

        else:
            if self.transform:
                img = self.transform(img)
            brain_embeddings_cls = self.Brain_embedding(torch.tensor(self.opened_memories, device=img.device))
            brain_embeddings_tsk = self.Brain_embedding_tsk(torch.tensor(self.opened_tasks, device=img.device))
            # brain_embeddings = torch.cat([brain_embeddings_tsk.repeat_interleave(self.cls_per_tsk, dim=0), brain_embeddings_cls], dim=-1)
            # brain_embeddings = brain_embeddings_cls
            #brain_bias = self.Brain_bias(torch.tensor(self.opened_memories, device=img.device))
            #C = brain_embeddings.shape[0]
            query_features = (self.query(x = img))#.unsqueeze(1).expand(-1, C, -1)
            # img_features = self.memory_map(img_features)#.mean(dim=1)
            # img_features = img_features
            img_features_tsk = self.embedding_map(query_features)
            #img_features = torch.cat([img_features_tsk, img_features_cls], dim=-1)
            # img_features = img_features_cls
            #img_features = self.all_map(img_features)
            #img_features = memory_gate + img_features + brain_bias
            #img_features = self.embedding_map(img_features)
            #brain_embeddings = self.vit(text=brain_embeddings.view(brain_embeddings.shape[0], -1, 768))
            #all_memories = self.embedding_map(brain_embeddings).mean(dim=0)
            #img_features = self.memory_map(torch.cat([img_features, all_memories.squeeze(0).repeat(img_features.shape[0],1)], dim=-1))
            if self.use_normalize:
                # img_features_cls = F.normalize(img_features_cls, dim=-1)
                # brain_embeddings_cls = F.normalize(brain_embeddings_cls, dim=-1)
                img_features_tsk = F.normalize(img_features_tsk, dim=-1)
                brain_embeddings_tsk = F.normalize(brain_embeddings_tsk, dim=-1)
                # img_features = F.normalize(img_features, dim=-1)
                # brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            #y_true = self.A(y_true)
            #y_brain_mem = self.A(y_brain_mem)
            task_info = torch.matmul(img_features_tsk, brain_embeddings_tsk.t())
            _, pred_task = torch.max(task_info, 1)
            # print(pred_task)
            # if task_info.shape[1]>=2:
            #     _, pred_task = task_info.topk(2, dim=1, largest=True)
            # else:
            #_, pred_task = torch.max(task_info, 1)
            img_features_cls = self.memory_map(self.vit(img, pred_task))
            if self.use_normalize:
                img_features_cls = F.normalize(img_features_cls, dim=-1)
                brain_embeddings_cls = F.normalize(brain_embeddings_cls, dim=-1)
            # logit_brain_mem = torch.matmul(img_features, brain_embeddings.t())
            logit_brain_mem = torch.matmul(img_features_cls, brain_embeddings_cls.t())
            #logit_brain_mem = self.mask_with_task(pred_task, logit_brain_mem)
            #logit_brain_mem = img_features * brain_embeddings.unsqueeze(0)
            return logit_brain_mem #.sum(dim=-1)'''


class Brain_Net_Vit(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        hidden_dim,
        cls_per_tsk,
        dropout=0.0,
        frozen_head=False,
        SN=False,  # SpectralNorm
        cosine_classifier=False,
        use_normalize = True,
        init="kaiming",
        device="cuda",
        use_bias=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.class_info = 1024 - hidden_dim
        self.image_size = image_size
        if image_size != 224:
            self.transform = nn.Sequential(
            Resize(size=(224, 224)),
            Normalize(torch.FloatTensor((0.5, 0.5, 0.5)), torch.FloatTensor((0.5, 0.5, 0.5)))
        )
        else:
            self.transform = False
        self.num_classes = num_classes
        self.use_normalize = use_normalize
        self.dropout = dropout
        self.frozen_head = frozen_head
        self.SN = SN
        self.init = init
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_bias = use_bias
        self.LN = nn.LayerNorm(512)
        
        #self.vit.parameters().requires_grad = False
        #for param in self.vit.parameters():
        #    param.requires_grad = False
        self.opened_memories = []
        self.opened_tasks = []
        self.memory_map = self._gen_projector(768, 3072, hidden_dim)
        #self.history_map = self._gen_projector(512, hidden_dim)
        self.task_num = int(num_classes/cls_per_tsk)
        self.query = Brain_Vit_att()
        self.vit =  None
        # 归一化向量
        #F.normalize(vectors, dim=-1)
        #print(torch.matmul(normalized_vectors, normalized_vectors.t()))

        # Create the embedding layer with the orthogonal matrix

        self.Brain_embedding = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.Brain_embedding.weight, mean=0, std=1)
        # nn.init.uniform_(self.Brain_embedding.weight, -1, 1)
        self.memory_2btrain = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.memory_2btrain.weight, mean=0, std=1)
        # nn.init.uniform_(self.memory_2btrain.weight, -1, 1)
        self.Brain_embedding_tsk = nn.Embedding(self.task_num, self.class_info)
        nn.init.normal_(self.Brain_embedding_tsk.weight, mean=0, std=1)
        # nn.init.uniform_(self.Brain_embedding_tsk.weight, -1, 1)
        self.memory_2btrain_tsk = nn.Embedding(self.task_num, self.class_info)
        nn.init.normal_(self.memory_2btrain_tsk.weight, mean=0, std=1)
        
        # nn.init.uniform_(self.memory_2btrain_tsk.weight, -1, 1)
        #print(torch.eye(768)[num_classes, :].shape)
        #self.memory_2btrain_tsk = nn.Embedding.from_pretrained(torch.eye(768)[:num_classes, :])
        self.external_att = SelfAttentionMLP(768, hidden_dim, 768, 768, self.class_info)
        self.external_att.apply(self.initialize_weights)
        self.task_now = 0
        self.cls_per_tsk = cls_per_tsk
        #self.Brain_embedding = nn.Embedding(num_classes, 512)
        #self.memory_2btrain = nn.Embedding(num_classes, 512)
        self.Brain_embedding.requires_grad = False
        self.Brain_embedding_tsk.requires_grad = False
        self.embedding_map = self._gen_projector(768, 3072, 256)
    
    def label_task(self, labels):
        tasks = torch.zeros_like(labels)
        bin_num = self.num_classes//self.cls_per_tsk
        for i in range(bin_num):
            bin_start = i * self.cls_per_tsk
            bin_end = (i+1) * self.cls_per_tsk
            tasks[(labels >= bin_start) & (labels < bin_end)] = i
        return tasks

    def check_memories(self, labels):  
        for label in labels.unique():
            if label.item() not in self.opened_memories:
                self.opened_memories.append(label.item())
        self.opened_memories.sort()

    def check_tasks(self, tasks):  
        for task in tasks.unique():
            if task.item() not in self.opened_tasks:
                self.opened_tasks.append(task.item())
        self.opened_tasks.sort()

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gen_projector(self, in_features, hidden_dim, out_dim):
        #projector = nn.Linear(in_features, out_dim, bias = False)
        projector = nn.Sequential(nn.Linear(in_features, hidden_dim, bias=self.use_bias), nn.GELU(), nn.Linear(hidden_dim, out_dim, bias=self.use_bias))
        #projector = ResidualLinear(in_features, hidden_dim, out_dim)
        projector.apply(self.initialize_weights)
        return projector
    
    def mask_with_task(self, pred_task, logits):
        mask = torch.zeros_like(logits)
        for i in range(pred_task.shape[0]):
            mask[i][pred_task[i]*self.cls_per_tsk:(pred_task[i]+1)*self.cls_per_tsk]=1
        masked_logits = mask * logits
        return masked_logits

    def forward(self, img, labels=None):

        if labels is not None:
            self.check_memories(labels)
            if self.transform:
                img = self.transform(img)
            
            condition = labels >= self.task_now * self.cls_per_tsk
            brain_embeddings_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels), self.Brain_embedding(labels))
            tsk = self.label_task(labels)
            self.check_tasks(tsk)
            condition = tsk >= self.task_now
            brain_embeddings_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(tsk), self.Brain_embedding_tsk(tsk))
            brain_embeddings = torch.cat([brain_embeddings_tsk, brain_embeddings_cls], dim=-1)
            img_features = self.query(img)
            img_features_cls = self.memory_map(img_features)
            
            # img_features_tsk = self.embedding_map(img_features)
            # img_features_cross = self.external_att(img_features, img_features_cls, img_features_tsk)
            img_features_tsk = self.external_att(img_features, img_features_cls, img_features)
            img_features = torch.cat([img_features_tsk, img_features_cls], dim=-1) #+ att
            
            # brain_embeddings = self.vit(text=brain_embeddings.view(brain_embeddings.shape[0], -1, 768))
            labels_all = torch.tensor(self.opened_memories, device=img.device)
            condition = labels_all >= self.task_now * self.cls_per_tsk
            all_memories_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels_all), self.Brain_embedding(labels_all))
            task_all = torch.tensor(self.opened_tasks, device=img.device)
            condition = task_all >= self.task_now
            all_memories_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(task_all), self.Brain_embedding_tsk(task_all))
            #print(torch.cat([torch.tensor([self.cls_per_tsk]*self.task_now), torch.tensor([all_memories_cls.shape[0]])], dim=0).int().to(all_memories_tsk.device), all_memories_cls.shape)
            if all_memories_cls.shape[0] >= self.cls_per_tsk * (self.task_now+1):
                all_memories = torch.cat([all_memories_tsk.repeat_interleave(self.cls_per_tsk, dim=0), all_memories_cls], dim=-1)
            else: 
                all_memories = torch.cat([all_memories_tsk.repeat_interleave(torch.cat([torch.tensor([self.cls_per_tsk]*self.task_now), torch.tensor([all_memories_cls.shape[0]%self.cls_per_tsk])], dim=0).int().to(all_memories_tsk.device), dim=0), all_memories_cls], dim=-1)
            #all_memories = all_memories_cls
            #all_memories_mean = self.embedding_map(all_memories).mean(dim=0)
            #img_features = self.embedding_map(img_features) #.mean(dim=1)
            #img_features = self.embedding_map(img_features)
            #img_features = self.memory_map(torch.cat([img_features, all_memories_mean.squeeze(0).repeat(img_features.shape[0],1)], dim=-1))
            #x_brain_mem = self.embedding_map(brain_embeddings)

            if self.use_normalize:
                img_features = F.normalize(img_features.mean(dim=1), dim=-1)
                brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            #y_true = self.A(y_true)
            #y_brain_mem = self.A(y_brain_mem)
            #y_history = self.A(y_history)
            logit_brain_mem = torch.matmul(img_features, brain_embeddings.t())
            #logit_embedding_mem = (torch.matmul(img_features_tsk, brain_embeddings_tsk.t()), tsk)
            #print(logit_brain_mem_tsk)
            return logit_brain_mem, None, F.normalize(all_memories, dim=-1)

        else:
            if self.transform:
                img = self.transform(img)
            brain_embeddings_cls = self.Brain_embedding(torch.tensor(self.opened_memories, device=img.device))
            brain_embeddings_tsk = self.Brain_embedding_tsk(torch.tensor(self.opened_tasks, device=img.device))
            brain_embeddings = torch.cat([brain_embeddings_tsk.repeat_interleave(self.cls_per_tsk, dim=0), brain_embeddings_cls], dim=-1)
            img_features = (self.query(x = img))#.unsqueeze(1).expand(-1, C, -1)
            img_features_cls = self.memory_map(img_features)#.mean(dim=1)
            # img_features = img_features
            # img_features_tsk = self.embedding_map(img_features)
            img_features_tsk = self.external_att(img_features, img_features_cls, img_features)
            # img_features_cross = self.external_att(img_features, img_features_cls, img_features_tsk)
            #att = self.external_att(img_features, img_features_cls, img_features_tsk)
            img_features = torch.cat([img_features_tsk, img_features_cls], dim=-1) #+ att
            
            if self.use_normalize:
                img_features = F.normalize(img_features.mean(dim=1), dim=-1)
                brain_embeddings = F.normalize(brain_embeddings, dim=-1)

            # logit_brain_mem = torch.matmul(img_features, brain_embeddings.t())
            logit_brain_mem = torch.matmul(img_features, brain_embeddings.t())
            return logit_brain_mem #.sum(dim=-1)


class SelfAttentionMLP(nn.Module):
    def __init__(self, input_dim0, input_dim1, input_dim2, hidden_dim, output_dim, num_heads=8):
        super(SelfAttentionMLP, self).__init__()
        self.pre_ori = nn.Sequential(
            nn.Identity()
            # nn.Linear(input_dim0, hidden_dim),
            # nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim)
        )
        self.pre_cls = nn.Sequential(
            # nn.Identity()
            nn.Linear(input_dim1, hidden_dim),
            # nn.GELU(),
            # nn.Linear(hidden_dim * expansion, hidden_dim)
        )
        self.pre_tsk = nn.Sequential(
            # nn.Identity()
            nn.Linear(input_dim2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        #self.norm = nn.LayerNorm(hidden_dim)
        # self.norm3 = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_ori, x_cls, x_tsk):

        x_ori = self.pre_ori(x_ori)
        x_cls = self.pre_cls(x_cls)
        x_tsk = self.pre_tsk(x_tsk)
        #cls 去query比较好
        #cls, ori, tsk: 76.78; 94.00
        #ori, cls, tsk: 75.67; 93.29
        attn_output, _ = self.attention(x_cls, x_ori, x_tsk)
        
        return self.out(attn_output)

class ResidualLinear(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_bias=True):
        super(ResidualLinear, self).__init__()
        self.use_bias = use_bias

        self.fc1 = nn.Linear(in_dim, in_dim, bias=self.use_bias)
        self.norm1 = nn.LayerNorm(in_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=self.use_bias)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=self.use_bias)

    def forward(self, x):
        
        out = self.fc1(x)
        out = self.act(out)
        out = self.norm1(out)+x

        out = self.fc2(out)
        out = self.act(out)

        out = self.fc3(out)
        
        return out
    

class MHMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, use_bias=True):
        super(MHMLP, self).__init__()
        self.use_bias = use_bias

        self.fc1 = MultiHeadLinear(in_dim, hidden_dim, num_heads, use_bias=self.use_bias)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = MultiHeadLinear(hidden_dim, hidden_dim, num_heads, use_bias=self.use_bias)
        self.fc3 = MultiHeadLinear(hidden_dim, out_dim, num_heads, use_bias=self.use_bias)

    def forward(self, x):
        
        out = self.fc1(x)
        out = self.act(out)
        out = self.norm1(out)

        out = self.fc2(out)
        out = self.act(out)

        out = self.fc3(out)
        
        return out

class MultiHeadLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, use_bias = True):
        super(MultiHeadLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.head_input_dim = input_dim // num_heads
        self.head_output_dim = output_dim // num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(self.head_input_dim, self.head_output_dim, bias = use_bias) for _ in range(num_heads)])

    def forward(self, x):
        input_splits = torch.chunk(x, self.num_heads, dim=1)

        head_outputs = [linear(input_split) for linear, input_split in zip(self.linear_layers, input_splits)]

        concatenated_output = torch.cat(head_outputs, dim=1)

        return concatenated_output

class Brain_Net(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        hidden_dim,
        dropout=0.0,
        cnnbackbone="ResNet18Pre",
        independent_classifier=False,
        frozen_head=False,
        SN=False,  # SpectralNorm
        cosine_classifier=False,
        use_normalize = True,
        init="kaiming",
        device="cuda",
        use_bias=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_normalize = use_normalize
        self.dropout = dropout
        self.cnnbackbone = cnnbackbone
        self.nf = 64 if image_size < 100 else 64
        self.independent_classifier = independent_classifier
        self.frozen_head = frozen_head
        self.SN = SN
        self.init = init
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_bias = use_bias
        self.pre_conv = nn.Identity()#nn.Sequential(conv3x3(3, 3, 1), nn.Sigmoid())
        self.conv = ResNet18_Brain()
        #self.mem_conv = ResNet18_Brain()

        self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), Rearrange("... () () -> ...")
            )

        '''self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])'''
        #self.brain = StableDiffusionPipeline.from_pretrained("/home/bqqi/.cache/huggingface/transformers/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
        #self.toprompt = label2prompt()
        #self.brain_memory = self.brain_memory_initialize()
        self.opened_memories = []
        self.memory_map = self._gen_projector(512, hidden_dim)

        self.mlp_head = self._gen_projector(512, hidden_dim)

        self.history_map = self._gen_projector(512, hidden_dim)
        self.Brain_embedding = nn.Embedding(num_classes, 768)
        self.embedding_map = self._gen_projector(768, 512)
        #self.A = nn.Linear(hidden_dim, hidden_dim, bias=False, requires_grad=False)
        
    def check_memories(self, labels):  
        for label in labels.unique():
            if label.item() not in self.opened_memories:
                self.opened_memories.append(label.item())
        self.opened_memories.sort()

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    '''def brain_memory_initialize(self):
        print("set initial brain memory")
        targets_all = torch.arange(start=0, end=self.num_classes)
        prompts_all = self.toprompt.map_labels_to_prompts(label_tensor = targets_all)
        with torch.no_grad():
            images = self.brain(prompts_all, num_inference_steps=50, eta=0.3, guidance_scale=6).images
        resized_images = [self.transform(image) for image in images]
        images = torch.stack(resized_images)
        self.brain_memory = images'''

    '''def update_brain_memory(self):
        print("brain memory update")
        targets_all = torch.arange(start=0, end=self.num_classes)
        prompts_all = self.toprompt.map_labels_to_prompts(label_tensor = targets_all)
        with torch.no_grad():
            images = self.brain(prompts_all, num_inference_steps=50, eta=0.3, guidance_scale=6).images
        resized_images = [self.transform(image) for image in images]
        images = torch.stack(resized_images)
        self.brain_memory = images'''

    def _gen_projector(self, in_features, hidden_dim):

        projector = nn.Sequential(nn.ReLU(), nn.Linear(in_features, in_features, bias=self.use_bias), nn.ReLU(), nn.Linear(in_features, hidden_dim, bias=self.use_bias))
        projector.apply(self.initialize_weights)
        return projector

    def forward(self, img, labels=None, mem=None):

        if labels is not None:
            self.check_memories(labels)
            img, img_brain_mem = img[:,0,:,:,:], img[:,1,:,:,:]
            x_brain_mem, x = self.conv(self.pre_conv(img_brain_mem)), self.conv(img)#, self.mem_conv(img)
            
            x_brain_mem, x = self.pool(x_brain_mem), self.pool(x)#, self.pool(x_comp)
            brain_embeddings = self.Brain_embedding(labels)
            x_brain_mem = x_brain_mem + self.embedding_map(brain_embeddings)
            y_brain_mem = self.memory_map(x_brain_mem)
            y_history = self.memory_map(x)
            y_true = self.mlp_head(x)
            if self.use_normalize:
                y_brain_mem = F.normalize(y_brain_mem, dim=-1)
                y_history = F.normalize(y_history, dim=-1)
                y_true = F.normalize(y_true, dim=-1)
            #y_true = self.A(y_true)
            #y_brain_mem = self.A(y_brain_mem)
            #y_history = self.A(y_history)
            logit_brain_mem = torch.matmul(y_true,y_brain_mem.t())
            logit_history = torch.matmul(y_true,y_history.t())
            return logit_brain_mem, logit_history, y_history, y_brain_mem

        else:
            img_brain_mem, img = mem, img
            x_brain_mem, x = self.conv(self.pre_conv(img_brain_mem)), self.conv(img)
            x_brain_mem, x = self.pool(x_brain_mem), self.pool(x)
            brain_embeddings = self.Brain_embedding(torch.tensor(self.opened_memories, device=x_brain_mem.device))
            x_brain_mem = x_brain_mem + self.embedding_map(brain_embeddings)
            y_brain_mem = self.memory_map(x_brain_mem)
            y_true = self.mlp_head(x)
            if self.use_normalize:
                y_brain_mem = F.normalize(y_brain_mem, dim=-1)
                y_true = F.normalize(y_true, dim=-1)
            #y_true = self.A(y_true)
            #y_brain_mem = self.A(y_brain_mem)
            logit = torch.matmul(y_true,y_brain_mem.t())
            return logit
