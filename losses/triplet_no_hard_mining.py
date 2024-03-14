from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb


class TripletLossNoHardMining(nn.Module):
    def __init__(self, margin=0, num_instances=8):
        super(TripletLossNoHardMining, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            for j in range(self.num_instances - 1):
                tmp = dist[i][mask[i]]
                dist_ap.append(tmp[j + 1])
                tmp = dist[i][mask[i] == 0]
                dist_an.append(tmp[j + 1])
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
