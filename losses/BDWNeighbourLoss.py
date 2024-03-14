from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from .DistWeightNeighbourLoss import DistWeightNeighbourLoss
import numpy as np


class BDWNeighbourLoss(nn.Module):
    def __init__(self, margin=0.7, slice=[0, 170, 341, 512]):
        super(BDWNeighbourLoss, self).__init__()
        self.s = slice
        self.margin = margin

    def forward(self, inputs, targets):
        inputs = [inputs[:, self.s[i] : self.s[i + 1]] for i in range(len(self.s) - 1)]
        loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        for input in inputs:
            loss, prec, pos_d, neg_d = DistWeightNeighbourLoss(margin=self.margin)(
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
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(BDWNeighbourLoss(margin=1)(inputs, targets))


if __name__ == "__main__":
    main()
    print("Congratulations to you!")
