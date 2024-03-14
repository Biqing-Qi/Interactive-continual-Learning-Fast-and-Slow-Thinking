from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class NeighbourLoss(nn.Module):
    # It is actually the online version LMNN
    def __init__(self, k=1, margin=0.1):
        super(NeighbourLoss, self).__init__()
        self.k = k
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist_mat = euclidean_dist(inputs)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist) // n + 1
        num_neg_instances = n - num_instances

        pos_dist = pos_dist.resize(
            len(pos_dist) // (num_instances - 1), num_instances - 1
        )
        neg_dist = neg_dist.resize(
            len(neg_dist) // num_neg_instances, num_neg_instances
        )

        #  clear way to compute the loss first
        loss = list()
        err = 0

        for i, pos_pair in enumerate(pos_dist):
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_dist[i])[0]
            pos_pair = pos_pair[: self.k]

            neg_pair = torch.masked_select(
                neg_pair, neg_pair < pos_pair[-1] + self.margin
            )

            if len(neg_pair) > 0:
                if i == 1 and np.random.randint(99) == 1:
                    # and np.random.randint(256) == 1:
                    print("neg_pair is ---------", neg_pair.data)
                    print("pos_pair is ---------", pos_pair.data)

                loss.append(torch.mean(pos_pair) - torch.mean(neg_pair) + self.margin)
                err += 1
            else:
                continue

        if len(loss) == 0:
            loss = 0.0 * (torch.mean(pos_pair))
        else:
            loss = torch.sum(torch.cat(loss)) / n

        prec = 1 - float(err) / n
        neg_d = torch.mean(neg_dist).data[0]
        pos_d = torch.mean(pos_dist).data[0]

        return loss, prec, pos_d, neg_d
