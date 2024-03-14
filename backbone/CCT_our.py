import torch
import torch.nn as nn
from .cvt_online import CVT_online, Brain_Net, Brain_Net_Vit, Brain_Vit_att


class CVT(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, image_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(CVT, self).__init__()
        print(output_size)
        self.image_size = image_size
        self.output_size = output_size
        self.stage = 3
        if self.stage == 1:
            self.net = CVT_online(
                image_size=self.image_size,
                num_classes=self.output_size,
                stages=1,  # number of stages
                dim=(256),  # dimensions at each stage
                depth=1,  # transformer of depth 4 at each stage
                heads=(4),  # heads at each stage
                mlp_mult=1,
                cnnbackbone="ResNet18Pre",  # 'ResNet18Pre' 'ResNet164'
                # independent_classifier=False,  # False True
                # frozen_head=False,
                dropout=0.1,
            )
        elif self.stage == 2:
            self.net = CVT_online(
                image_size=self.image_size,
                num_classes=self.output_size,
                stages=2,  # number of stages
                dim=(256, 512),  # dimensions at each stage
                depth=2,  # 2 transformer of depth 4 at each stage
                heads=(4, 8),  # heads at each stage
                mlp_mult=2,  # 2
                cnnbackbone="ResNet18Pre",  # 'ResNet18Pre' 'ResNet164'
                dropout=0.1,
            )
        elif self.stage == 3:
            self.net = CVT_online(
                image_size=self.image_size,
                num_classes=self.output_size,
                stages=3,  # number of stages
                dim=(128, 256, 512),  # dimensions at each stage (128, 256, 512)
                depth=2,  # 2 transformer of depth 4 at each stage
                heads=(2, 4, 8),  # heads at each stage
                mlp_mult=2,  # 2
                cnnbackbone="ResNet18Pre",  # 'ResNet18Pre' 'ResNet164'
                dropout=0.1,
            )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (100)
        """
        x = self.net.conv(x)

        if hasattr(self.net, "transformers"):
            x = [transformer(x) for transformer in self.net.transformers]
            x = torch.stack(x).sum(dim=0)  # add growing transformers' output
            # x = torch.stack(x).mean(dim=0)
        else:
            x = self.net.transformer(x)

        x = self.net.pool(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = self.net(x)
        return x

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_key_bias_params(self):
        params = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                params.append(param.view(-1))
        return torch.cat(params)

    def get_key_bias_grad(self):
        grads = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                grads.append(param.grad.data.view(-1))
        return torch.cat(grads)

    def frozen(self, t):
        if hasattr(self.net, "frozen"):
            self.net.frozen(t)


class Brain_Coworker(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, image_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(Brain_Coworker, self).__init__()
        self.image_size = image_size
        self.output_size = output_size

        self.net = Brain_Net(
                image_size=self.image_size,
                num_classes=self.output_size,
                hidden_dim=128,
                dropout=0.1,
            )
        print('Brain_Co loaded done')
    def forward(self, x, labels=None, mem=None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        if labels is not None:
            logit_brain_mem, logit_history, y_history, y_brain_mem = self.net(x, labels)
            return logit_brain_mem, logit_history, y_history, y_brain_mem
        else:
            logit = self.net(x, labels, mem)
            return logit

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_key_bias_params(self):
        params = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                params.append(param.view(-1))
        return torch.cat(params)

    def get_key_bias_grad(self):
        grads = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                grads.append(param.grad.data.view(-1))
        return torch.cat(grads)

    def frozen(self, t):
        if hasattr(self.net, "frozen"):
            self.net.frozen(t)

class Brain_Coworker_Vit(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, image_size: int, output_size: int, hidden_dim: int, cls_per_tsk: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(Brain_Coworker_Vit, self).__init__()
        self.image_size = image_size
        self.output_size = output_size

        self.net = Brain_Net_Vit(
                image_size=self.image_size,
                num_classes=self.output_size,
                hidden_dim=hidden_dim,
                cls_per_tsk=cls_per_tsk,
                dropout=0.1,
            )
        
    def forward(self, x, labels=None, mem=None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        if labels is not None:
            logit_brain_mem_cls, logits, all_memories = self.net(x, labels)
            return logit_brain_mem_cls, logits, all_memories
        else:
            logit = self.net(x)
            return logit

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_key_bias_params(self):
        params = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                params.append(param.view(-1))
        return torch.cat(params)

    def get_key_bias_grad(self):
        grads = []
        for name, param in self.named_parameters():
            if "mk.0" in name or "pos_bias" in name:
                grads.append(param.grad.data.view(-1))
        return torch.cat(grads)

    def frozen(self, t):
        if hasattr(self.net, "frozen"):
            self.net.frozen(t)
            

class VitPre(nn.Module):
    def __init__(self, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(VitPre, self).__init__()
        # self.image_size = image_size
        self.output_size = output_size
        self.num_classes = output_size

        self.net = Brain_Vit_att()
        self.classifer = nn.Linear(768, output_size)
        
    def forward(self, x) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = self.net(x)
        x = torch.mean(x, dim=1)  # 对中间维度取平均
        x = torch.squeeze(x)  # 去掉中间维度 
        logits = self.classifer(x)
        
        return logits