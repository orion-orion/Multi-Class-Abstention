# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(HingeLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, input):
        return self.c * torch.clamp(1 - self.alpha * input, min=0)


class ExponentialLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, input):
        return self.c * torch.exp(-self.alpha * input)


class LogisticLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(LogisticLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, input):
        return self.c * torch.log1p(torch.exp(-self.alpha * input))


class LabelPred(nn.Module):
    def __init__(self):
        super(LabelPred, self).__init__()

    def forward(self, preds, y):
        return preds[y]


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input1, input2):
        return 1 - (torch.exp(input1)
                    / torch.exp(input2).sum(dim=-1)).view(-1, 1)


class MarginLoss(nn.Module):
    def __init__(self, alpha=1):
        super(MarginLoss, self).__init__()
        self.hinge_loss = HingeLoss(alpha)

    def forward(self, input):
        return torch.clamp(self.hinge_loss(input), max=1)


class PredictionMargin(nn.Module):
    def __init__(self):
        super(PredictionMargin, self).__init__()

    def forward(self, input_1, input_2, excl_idx=None):
        return input_1.view(-1, 1) - torch.max(
            input_2[~excl_idx].view(input_2.shape[0], -1), dim=-1)[0]


class PairwiseDiff(nn.Module):
    def __init__(self):
        super(PairwiseDiff, self).__init__()

    def forward(self, input_1, input_2, excl_idx=None):
        if excl_idx is not None:
            pairwise_diff_matrix = \
                input_1.view(-1, 1) - \
                input_2[~excl_idx].view(input_2.shape[0], -1)
        else:
            pairwise_diff_matrix = (input_1.view(-1, 1) - input_2)
        return pairwise_diff_matrix
