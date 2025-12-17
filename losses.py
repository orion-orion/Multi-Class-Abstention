# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from loss_modules import HingeLoss, PredictionMargin, ExponentialLoss, \
    LogisticLoss, PairwiseDiff, LabelPred, MAELoss, MarginLoss
import logging
import torch.nn.functional as F


class OurLoss(nn.Module):
    def __init__(self, surr_type, phi_type, psi_type, alpha=1, c=0.1,
                 if_normalize=False):
        super(OurLoss, self).__init__()
        self.surr_type = surr_type
        self.phi_type = phi_type
        self.psi_type = psi_type
        self.label_pred = LabelPred()
        self.ce = nn.CrossEntropyLoss()
        self.if_normalize = if_normalize
        if self.phi_type == "exponential":
            self.phi = ExponentialLoss()
        elif self.phi_type == "hinge":
            self.phi = HingeLoss()
        elif self.phi_type == "logistic":
            self.phi = LogisticLoss()
        else:
            raise Exception(
                "phi should be one of exponential, hinge or logistic")
        if self.surr_type == "MCS":
            if self.psi_type == "exponential":
                self.psi_1 = ExponentialLoss(alpha=-1)
                self.psi_2 = ExponentialLoss(
                    alpha, c)
            elif self.psi_type == "hinge":
                self.psi_1 = HingeLoss(alpha=-1)
                self.psi_2 = HingeLoss(
                    alpha, c)
            elif self.psi_type == "logistic":
                self.psi_1 = LogisticLoss(alpha=-1)
                self.psi_2 = LogisticLoss(
                    alpha, c)
            else:
                raise Exception(
                    "psi1, psi2 should be one of exponential, hinge or"
                    "logistic")
        elif self.surr_type == "ACS":
            if self.psi_type == "exponential":
                self.psi = ExponentialLoss(alpha, c)
            elif self.psi_type == "hinge":
                self.psi = HingeLoss(alpha, c)
            elif self.psi_type == "logistic":
                self.psi = LogisticLoss(alpha, c)
            else:
                raise Exception(
                    "psi should be one of exponential, hinge or logistic")
        else:
            raise Exception(
                "Method should be one of MCS and ACS")
        self.prediction_margin = PredictionMargin()

    def forward(self, preds, rej_scores, y):
        if self.if_normalize:
            # For numerical stability
            if not torch.all((preds >= -10) * (preds <= 10)):
                preds = F.normalize(preds, dim=-1)
            # For numerical stability
            if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
                rej_scores = F.normalize(rej_scores, dim=0)
            # if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
            #     rej_scores = 2 * F.sigmoid(rej_scores) - 1

        preds_y = self.label_pred(preds, y)
        margin = self.prediction_margin(preds_y, preds, y)
        if self.surr_type == "MCS":
            # loss = (self.phi(margin) * self.psi_1(rej_scores)).mean() \
            #     + self.psi_2(rej_scores).mean()
            loss = self.phi(margin).mean()

        elif self.surr_type == "ACS":
            loss = self.phi(margin - rej_scores).mean() \
                + self.psi(rej_scores).mean()
        else:
            raise Exception(
                "Method should be one of MCS and ACS")
        return loss

    def forward_stage1(self, preds, y):
        return self.ce(preds, y)

    def forward_stage2(self, preds, rej_scores, y):
        if self.psi_type == "exponential":
            # For numerical stability
            if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
                rej_scores = F.normalize(rej_scores, dim=0)

        _, pred_y = torch.max(preds.data, 1)

        zero_one_clf_loss = (pred_y == y)

        loss = (zero_one_clf_loss * self.psi_1(rej_scores)).mean() \
            + self.psi_2(rej_scores).mean()

        return loss


class MaoLoss(nn.Module):
    def __init__(self, l_type, psi_type="exponential", alpha=1, c=0.05, if_normalize=False):
        super(MaoLoss, self).__init__()
        self.l_type = l_type
        self.psi_type = psi_type
        self.alpha, self.c = alpha, c
        self.label_pred = LabelPred()
        self.ce = nn.CrossEntropyLoss()
        self.if_normalize = if_normalize
        if self.l_type == "MAE":
            self.mae = MAELoss()
        elif self.l_type == "C-Hinge":
            self.phi = HingeLoss(alpha)
            self.pw_diff = PairwiseDiff()
        elif self.l_type == "Margin":
            self.phi = MarginLoss(alpha)
            self.prediction_margin = PredictionMargin()
        if self.psi_type == "exponential":
            self.psi_1 = ExponentialLoss(alpha=-1)
            self.psi_2 = ExponentialLoss(alpha, c)
        elif self.psi_type == "hinge":
            self.psi_1 = HingeLoss(alpha=-1)
            self.psi_2 = HingeLoss(alpha, c)
        elif self.psi_type == "logistic":
            self.psi_1 = LogisticLoss(alpha=-1)
            self.psi_2 = LogisticLoss(alpha, c)
        else:
            raise Exception(
                "psi1, psi2 should be one of exponential, hinge or"
                "logistic")

    def forward(self, preds, rej_scores, y):
        if self.if_normalize:
            if not torch.all((preds >= -10) * (preds <= 10)):
                preds = F.normalize(preds, dim=-1)
            # For numerical stability
            if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
                rej_scores = F.normalize(rej_scores, dim=0)

        lam = torch.rand(preds.shape[0], 1)
        preds_y = self.label_pred(preds, y)
        if self.l_type == "MAE":
            mul_cls_l = self.mae(preds_y, preds)

        elif self.l_type == "C-Hinge":
            mul_cls_l = self.phi(self.pw_diff(0, preds, y)).sum(dim=-1) \
                - lam * preds.sum(dim=-1)

        elif self.l_type == "Margin":
            margin = self.prediction_margin(preds_y, preds, y)
            mul_cls_l = self.phi(margin)

        loss = (mul_cls_l * self.psi_1(rej_scores)).mean() \
            + self.psi_2(rej_scores).mean()

        return loss

    def forward_stage1(self, preds, y):
        return self.ce(preds, y)

    def forward_stage2(self, preds, rej_scores, y):
        # For numerical stability
        if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
            rej_scores = F.normalize(rej_scores, dim=0)

        _, pred_y = torch.max(preds.data, 1)

        zero_one_clf_loss = (pred_y != y).float().view(-1, 1)

        loss = (zero_one_clf_loss * self.psi_1(rej_scores)).mean() \
            + self.psi_2(rej_scores).mean()

        return loss


class NiLoss(nn.Module):
    def __init__(self, surr_type, phi_type, psi_type="exponential", alpha=1,
                 c=0.01, if_normalize=False):
        super(NiLoss, self).__init__()
        self.surr_type = surr_type
        self.psi_type = psi_type
        self.alpha, self.c = alpha, c
        self.surr_type = surr_type
        self.phi_type = phi_type
        self.if_normalize = if_normalize
        if self.phi_type == "exponential":
            self.phi = ExponentialLoss()
        elif self.phi_type == "hinge":
            self.phi = HingeLoss()
        elif self.phi_type == "logistic":
            self.phi = LogisticLoss()
        else:
            raise Exception(
                "phi should be one of exponential, hinge or logistic")
        if self.surr_type == "MCS":
            self.pw_diff = PairwiseDiff()
            self.label_pred = LabelPred()
            if self.psi_type == "exponential":
                self.psi_1 = ExponentialLoss(alpha=-1)
                self.psi_2 = ExponentialLoss(alpha, c)
            elif self.psi_type == "hinge":
                self.psi_1 = HingeLoss(alpha=-1)
                self.psi_2 = HingeLoss(alpha, c)
            elif self.psi_type == "logistic":
                self.psi_1 = LogisticLoss(alpha=-1)
                self.psi_2 = LogisticLoss(alpha, c)
            else:
                raise Exception(
                    "psi1, psi2 should be one of exponential, hinge or"
                    "logistic")
        elif self.surr_type == "ACS":
            self.pw_diff = PairwiseDiff()
            self.label_pred = LabelPred()
            if self.psi_type == "exponential":
                self.psi = ExponentialLoss(alpha, c)
            elif self.psi_type == "hinge":
                self.psi = HingeLoss(alpha, c)
            elif self.psi_type == "logistic":
                self.psi = LogisticLoss(alpha, c)
            else:
                raise Exception(
                    "psi should be one of exponential, hinge or logistic")

        elif self.surr_type == "OVA":
            self.pw_diff = PairwiseDiff()
            self.label_pred = LabelPred()

        elif self.surr_type == "CE":
            self.ce = nn.CrossEntropyLoss()

        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")

    def forward(self, preds, rej_scores, y):
        if self.if_normalize and self.surr_type in ["MCS", "ACS", "OVA"]:
            # For numerical stability
            if not torch.all((preds >= -10) * (preds <= 10)):
                preds = F.normalize(preds, dim=-1)
        if self.if_normalize and self.surr_type in ["MCS", "ACS"]:
            # For numerical stability
            if not torch.all((rej_scores >= -1) * (rej_scores <= 1)):
                rej_scores = F.normalize(rej_scores, dim=0)

        if self.surr_type == "MCS":
            preds_y = self.label_pred(preds, y)
            diff = self.pw_diff(preds_y, preds, y)
            loss = (self.phi(diff).sum(dim=1) * self.psi_1(rej_scores)).mean()\
                + self.psi_2(rej_scores).mean()

        elif self.surr_type == "ACS":
            preds_y = self.label_pred(preds, y)
            diff = self.pw_diff(preds_y, preds, y)
            loss = (self.phi(diff - rej_scores).sum(dim=-1)).mean()
            + self.psi(rej_scores).mean()

        elif self.surr_type == "OVA":
            preds_y = self.label_pred(preds, y)
            loss = self.phi(preds_y).mean() \
                + self.phi(self.pw_diff(0, preds, y)).sum(dim=-1).mean()

        elif self.surr_type == "CE":
            loss = self.ce(preds, y)
        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")
        return loss
