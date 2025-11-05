# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from loss_modules import HingeLoss, PredictionMargin, ExponentialLoss, \
    LogisticLoss, PairwiseDiff, LabelPred, MAELoss, MarginLoss


class OurLoss(nn.Module):
    def __init__(self, surr_type, psi_type, alpha=1, c=1):
        super(OurLoss, self).__init__()
        self.surr_type = surr_type
        self.psi_type = psi_type
        self.alpha, self.c = alpha, c
        self.label_pred = LabelPred()
        if self.surr_type == "MCS":
            self.phi = ExponentialLoss()
            assert (self.psi_type == "exponential")
            self.psi_1 = ExponentialLoss()
            self.psi_2 = ExponentialLoss(
                self.alpha, self.c)
        else:
            self.phi = HingeLoss()
            assert (self.psi_type in ["exponential", "hinge"])
            if self.psi_type == "exponential":
                self.psi = ExponentialLoss(
                    self.alpha, self.c)
            else:
                self.psi = HingeLoss(self.alpha, self.c)
        self.prediction_margin = PredictionMargin()

    def forward(self, preds, rej_scores, y):
        preds_y = self.label_pred(preds, y)
        margin = self.prediction_margin(preds_y, preds, y)
        if self.surr_type == "MCS":
            loss = (self.phi(margin) * self.psi_1(rej_scores)).mean() \
                + self.psi_2(rej_scores).mean()

        elif self.surr_type == "ACS":
            loss = self.phi(margin - rej_scores).mean() \
                + self.psi_2(rej_scores).mean()
        else:
            raise Exception(
                "Method should be one of MCS and ACS")
        return loss


class MaoLoss(nn.Module):
    def __init__(self, l_type, psi_type="ExponentialLoss", alpha=1, c=1):
        super(OurLoss, self).__init__()
        self.l_type = l_type
        self.psi_type = psi_type
        self.alpha, self.c = alpha, c
        if self.l_type == "MAE":
            self.mae = MAELoss()
        elif self.l_type == "C-Hinge":
            self.phi = HingeLoss(alpha)
            self.pw_diff = PairwiseDiff()
        elif self.l_type == "Margin":
            self.label_pred = LabelPred()
            self.phi = MarginLoss(alpha)
            self.prediction_margin = PredictionMargin()
        if self.psi_type == "exponential":
            self.psi_1 = ExponentialLoss()
            self.psi_2 = ExponentialLoss(alpha, c)
        else:
            self.psi_1 = HingeLoss()
            self.psi_2 = HingeLoss(alpha, c)

    def forward(self, preds, rej_scores, y):
        lam = torch.rand(preds.shape[0], 1)
        if self.mao_l_type == "MAE":
            mul_cls_l = self.mae(preds)

        elif self.mao_l_type == "C-Hinge":
            mul_cls_l = self.phi(self.pw_diff(0, preds, y)).sum(-1) \
                - lam * preds.sum(-1)

        elif self.mao_l_type == "Margin":
            preds_y = self.label_pred(preds, y)
            margin = self.prediction_margin(preds_y, preds, y)
            mul_cls_l = self.phi(margin)

        loss = (mul_cls_l * self.psi_1(rej_scores)).mean() \
            + self.psi_2(rej_scores).mean()

        return loss


class NiLoss(nn.Module):
    def __init__(self, surr_type, psi_type="logistic", alpha=1, c=1):
        super(OurLoss, self).__init__()
        self.surr_type = surr_type
        self.psi_type = psi_type
        self.alpha, self.c = alpha, c
        self.surr_type = surr_type
        if self.surr_type == "MCS":
            self.pw_diff = PairwiseDiff().to(self.device)
            self.label_pred = LabelPred().to(self.device)
            self.phi = LogisticLoss().to(self.device)
            self.psi_1 = LogisticLoss().to(self.device)
            self.psi_2 = LogisticLoss(alpha=1, c=self.c).to(self.device)
        else:
            self.phi = LogisticLoss().to(self.device)
            self.psi = LogisticLoss(alpha=1, c=self.c).to(self.device)

    def forward(self, preds, rej_scores, y):
        preds_y = self.label_pred(preds, y)

        diff = self.pw_diff(preds_y, preds, y)
        if self.surr_type == "MCS":
            loss = (self.phi(diff).sum(
                1) * self.psi_1(rej_scores)).mean()\
                + self.psi_2(rej_scores).mean()

        elif self.surr_type == "ACS":
            loss = (self.phi(diff - rej_scores)
                    .sum(-1) * self.psi_1(rej_scores)).mean()
            + self.psi_2(rej_scores).mean()

        elif self.surr_type == "OVA":
            loss = self.phi(preds_y).mean() \
                + self.phi(self.pw_diff(0, preds, y)).sum(-1).mean()

        elif self.surr_type == "CE":
            loss = -preds_y.mean()
            + torch.log(torch.exp(preds)).sum(-1).mean()
        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")
        return loss
