from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from .DistWeightDevianceLoss import DistWeightBinDevianceLoss
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class DistWeightDevBranchLoss(nn.Module):
    def __init__(self, margin=0.5, position=[0, 170, 341, 512]):
        super(DistWeightDevBranchLoss, self).__init__()
        self.s = position
        self.margin = margin

    def forward(self, inputs, targets):
        inputs = [inputs[:, self.s[i] : self.s[i + 1]] for i in range(len(self.s) - 1)]
        loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        for input in inputs:
            loss, prec, pos_d, neg_d = DistWeightBinDevianceLoss(margin=self.margin)(
                input, targets
            )
            loss_list.append(loss)
            prec_list.append(prec)
            pos_d_list.append(pos_d)
            neg_d_list.append(neg_d)

        loss = torch.mean(torch.cat(loss_list))
        prec = np.mean(prec_list)
        pos_d = np.mean((pos_d_list))
        neg_d = np.mean((neg_d_list))

        return loss, prec, pos_d, neg_d


def main():
    data_size = 32
    input_dim = 8
    output_dim = 512
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(DistWeightDevBranchLoss()(inputs, targets))


if __name__ == "__main__":
    main()
    print("Congratulations to you!")
