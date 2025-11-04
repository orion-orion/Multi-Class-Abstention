# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, emb_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(emb_size * 2, 1)

    def forward(self, input1, input2):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1)
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1)
            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        input = torch.cat([smaller, larger], dim=-1)
        output = self.fc(input)
        return output


class HingeLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(HingeLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, input):
        return self.c * torch.clamp(1 - self.alpha * input, 0)


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
        return self.c * torch.log1p(self.alpha * input)


class LabelPred(nn.Module):
    def __init__(self):
        super(LabelPred).__init__()

    def forward(self, preds, y):
        return preds[y]


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
        if excl_idx:
            pairwise_diff_matrix = \
                input_1.view(-1, 1) - \
                input_2[~excl_idx].view(input_2.shape[0], -1)
        else:
            pairwise_diff_matrix = (input_1.view(-1, 1) - input_2)
        return pairwise_diff_matrix


class JSDLoss(torch.nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, pos, neg):
        pos = -F.softplus(-pos)
        neg = F.softplus(neg)
        return neg - pos


class BiDiscriminator(torch.nn.Module):
    def __init__(self, emb_size):
        super(BiDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(emb_size, emb_size, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, input1, input2, s_bias=None):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1)
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1)
            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        score = self.f_k(smaller, larger)
        if s_bias is not None:
            score += s_bias
        return score
