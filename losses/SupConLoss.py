from __future__ import print_function

import torch
import torch.nn as nn
import pdb


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(
        self, features, labels=None, mask=None, focuses=None, focus_labels=None
    ):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            focuses: learnable class prototype.
            focus_labels: prototype label.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if focus_labels is not None:
                # print(labels)
                # print(focus_labels)
                labels = torch.cat((labels, labels, focus_labels))
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size * 2 + len(focus_labels):
                    raise ValueError("Num of labels does not match num of features")
            else:
                labels = torch.cat((labels, labels))
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size * 2:
                    raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views 2
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # remove n_view, [2N, dims]
        # pdb.set_trace()
        if focus_labels is not None:
            contrast_feature = torch.cat([contrast_feature, focuses], dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # n_views 2
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))
        num_instance = anchor_feature.shape[0]
        # print(num_instance)
        # print(features.mean())
        # print(focuses.mean())

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )  # I, all z_ij, 2N

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = (
            anchor_dot_contrast - logits_max.detach()
        )  # avoid very large logits that lead to the NaN problem
        # logits = anchor_dot_contrast
        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # pdb.set_trace()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_instance).view(-1, 1).to(device),
            0,
        )  # A(i)

        logits_mask[:, batch_size * anchor_count :] = (
            logits_mask[:, batch_size * anchor_count :] * 10
        )
        mask = mask * logits_mask  # P(i)

        # pdb.set_trace()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 0.01)

        # loss
        loss = -1 * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        # print('loss: ', loss)
        return loss

        # focus_weight = torch.ones_like(labels)
        # focus_weight[batch_size * anchor_count:] = 10
        # focus_mask = torch.matmul(focus_weight, focus_weight.T).to(device)
        # logits_mask = focus_mask * logits_mask  # A(i) with weight
