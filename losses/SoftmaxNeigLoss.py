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


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


class SoftmaxNeigLoss(nn.Module):
    def __init__(self, alpha=50):
        super(SoftmaxNeigLoss, self).__init__()
        self.alpha = alpha
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

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
            # pos_pair = pos_pair[:4]
            pos_neig = pos_pair[:5]
            neg_neig = neg_pair[:15]

            if i == 1 and np.random.randint(64) == 1:
                print("pos_pair is ---------", pos_neig)
                print("neg_pair is ---------", neg_neig)

            base = 1.0
            pos_logit = torch.sum(torch.exp(self.alpha * (base - pos_pair)))
            neg_logit = torch.sum(torch.exp(self.alpha * (base - neg_pair))) / 2

            loss_ = -torch.log(pos_logit / (pos_logit + neg_logit))
            loss.append(loss_)

        loss = torch.mean(torch.cat(loss))

        prec = 1 - float(err) / n
        neg_d = torch.mean(neg_dist).data[0]
        pos_d = torch.mean(pos_dist).data[0]

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

    print(SoftmaxNeigLoss(margin=0.1)(inputs, targets))


if __name__ == "__main__":
    main()
    print("Congratulations to you!")
