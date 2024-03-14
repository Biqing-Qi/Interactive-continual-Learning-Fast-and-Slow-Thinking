import torch
from torch import nn


class MultiSimilarityLoss(nn.Module):
    """
    Base source code taken from the orig. implementation:
    https://github.com/MalongTech/research-ms-loss/
    """

    def __init__(
        self, thresh=0.5, _margin=0.1, scale_pos=2.0, scale_neg=40.0, **kwargs
    ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = _margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.epsilon = 1e-5

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(
            0
        ), "feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        loss = torch.tensor(0.0)
        if feats.is_cuda:
            loss = loss.cuda()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - self.epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = (
                1.0
                / self.scale_pos
                * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
                )
            )
            neg_loss = (
                1.0
                / self.scale_neg
                * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
                )
            )
            loss += pos_loss + neg_loss

        if loss == 0:
            return torch.zeros([], requires_grad=True)

        return loss / batch_size
