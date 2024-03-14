from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


def GaussDistribution(data):
    """

    :param data:
    :return:
    """
    mean_value = torch.mean(data).data[0]
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2))).data[0]
    return mean_value, std


class MarginDevianceLoss(nn.Module):
    def __init__(self):
        super(MarginDevianceLoss, self).__init__()

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = similarity(inputs)
        # print(sim_mat)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim) // n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim) // (num_instances - 1), num_instances - 1)
        neg_sim = neg_sim.resize(len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = list()
        c = 0

        # gauss is a numpy matrix to keep the gaussian mean and variance
        # and intersection point value

        gauss = np.zeros([n, 5])

        for i, pos_pair in enumerate(pos_sim):
            # print(i)
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_sim[i])[0]

            pos_mean, pos_std = GaussDistribution(pos_pair)
            neg_mean, neg_std = GaussDistribution(neg_pair)

            inter = (neg_std * pos_mean + pos_std * neg_mean) / (pos_std + neg_std)
            inter = 0.8 * inter + 0.1
            gauss[i] = [pos_mean, neg_mean, pos_std, neg_std, inter]
            neg_pair = torch.masked_select(neg_pair, neg_pair > pos_pair[0] - 0.05)
            # pos_pair = pos_pair[1:]
            if len(neg_pair) < 1:
                c += 1
                continue

            neg_pair = torch.sort(neg_pair)[0]

            if i == 1 and np.random.randint(199) == 1:
                print("neg_pair is ---------", neg_pair)
                print("pos_pair is ---------", pos_pair.data)
            pos_loss = 0.2 * torch.mean(
                torch.log(1 + torch.exp(-10 * (pos_pair - inter)))
            )

            neg_loss = 0.05 * torch.mean(
                torch.log(1 + torch.exp(40 * (neg_pair - inter)))
            )
            loss.append(pos_loss + neg_loss)
        print(gauss[1])
        loss = torch.sum(torch.cat(loss)) / n

        prec = float(c) / n
        neg_d = torch.mean(neg_sim).data[0]
        pos_d = torch.mean(pos_sim).data[0]

        return loss, prec, pos_d, neg_d


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(MarginDevianceLoss()(inputs, targets))


if __name__ == "__main__":
    main()
    print("Congratulations to you!")
