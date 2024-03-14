from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class DistanceMatchLoss(nn.Module):
    def __init__(self, margin=1):
        super(DistanceMatchLoss, self).__init__()
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
            # pos_mean, pos_std = GaussDistribution(pos_pair)
            # prob = torch.exp(torch.pow(pos_pair - pos_mean, 2) / (2 * torch.pow(pos_std, 2)))
            # pos_index = torch.multinomial(prob, 3, replacement=False)
            #
            # pos_pair = pos_pair[pos_index]
            #
            pos_pair = torch.sort(pos_pair)[0]
            pos_pair = pos_pair[:3]

            # sampling negative
            neg_pair = neg_dist[i]
            neg_mean, neg_std = GaussDistribution(neg_pair)
            prob = torch.exp(
                torch.pow(neg_pair - neg_mean, 2) / (2 * torch.pow(neg_std, 2))
            )
            neg_index = torch.multinomial(prob, 3 * num_instances, replacement=False)

            neg_pair = neg_pair[neg_index]

            neg_pair = torch.masked_select(neg_pair, neg_pair < pos_pair[2] + 0.05)

            if len(neg_pair) > 0:
                neg_pair = torch.sort(neg_pair)[0]

                if i == 1 and np.random.randint(99) == 1:
                    # and np.random.randint(256) == 1:
                    print("neg_pair is ---------", neg_pair)
                    print("pos_pair is ---------", pos_pair.data)

                # neg_base = torch.sum(torch.exp(-10*(neg_pair - 1))*neg_pair)/torch.sum(torch.exp(-10*(neg_pair - 1)))
                # base = 0.95/pos_pair[0].data[0]*pos_pair.data
                base = [0.95, 1.05, 1.12]
                muls = [4, 8, 16]
                pos_diff = [pos_pair[i] - base[i] for i in range(len(base))]
                pos_diff = torch.cat(
                    [
                        1.0 / muls[i] * torch.log(1 + torch.exp(pos_diff[i]))
                        for i in range(len(base))
                    ]
                )
                pos_loss = torch.mean(pos_diff)
                neg_loss = 0.02 * torch.mean(
                    torch.log(1 + torch.exp(50 * (self.margin - neg_pair)))
                )

                loss.append(pos_loss + neg_loss)

                if pos_pair[0].data[0] > neg_pair[0].data[0] - 0.05:
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


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    # print('training data is ', x)
    # print('initial parameters are ', w)
    inputs = x.mm(w)
    # print('extracted feature is :', inputs)

    # y_ = np.random.randint(num_class, size=data_size)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(DistanceMatchLoss(margin=1)(inputs, targets))


if __name__ == "__main__":
    main()
    print("Congratulations to you!")
