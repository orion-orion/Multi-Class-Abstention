# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.resnet import resnet1202
from models.hl_filter.hf_model import HF
from models.dhcf.dhcf_model import DHCF
from models.mf.mf_model import NeuMF
from models.gnn.gnn_model import GNN
from models.pricdr.pricdr_model import PriCDR
from models.p2fcdr.p2fcdr_model import P2FCDR
from models.ppdm.ppdm_model import PPDM
from utils.io_utils import ensure_dir
from utils import train_utils
from losses import HingeLoss, PredictionMargin, ExponentialLoss, \
    LogisticLoss, PairwiseDiff, LabelPred


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        if self.method == "FedHCDR":
            train_utils.change_lr(self.hi_optimizer, new_lr)
            train_utils.change_lr(self.lo_optimizer, new_lr)
        else:
            train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, data_info):
        self.args = args
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.predictor = resnet1202(in_channels=data_info["n_channels"],
                                    output_size=data_info["num_cls"])
        self.rejector = resnet1202(in_channels=data_info["n_channels"],
                                   output_size=1)
        self.c = args.c
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = (args.model_id if len(args.model_id)
                         > 1 else "0" + args.model_id)
        if self.method == "Ours":
            self.our_surr_type = args.our_surr_type
            if self.our_surr_type == "MCS":
                self.phi = ExponentialLoss().to(self.device)
                assert (self.psi_type == "exponential")
                self.psi_1 = ExponentialLoss().to(self.device)
                self.psi_2 = ExponentialLoss(
                    args.alpha, self.c).to(self.device)
            else:
                self.phi = HingeLoss.to(self.device)
                assert (self.psi_type in ["exponential", "hinge"])
                if self.psi_type == "exponential":
                    self.psi = ExponentialLoss(
                        args.alpha, self.c).to(self.device)
                else:
                    self.psi = HingeLoss(args.alpha, self.c).to(self.device)
            self.prediction_margin = PredictionMargin()
            self.bce_criterion = nn.BCEWithLogitsLoss().to(self.device)

        elif self.method == "Ni+":
            self.ni_et_al_surr_type = args.ni_et_al_surr_type
            if self.ni_et_al_surr_type == "MCS":
                self.pw_diff = PairwiseDiff().to(self.device)
                self.label_pred = LabelPred().to(self.device)
                self.phi = LogisticLoss().to(self.device)
                self.psi_1 = LogisticLoss().to(self.device)
                self.psi_2 = LogisticLoss(alpha=1, c=self.c).to(self.device)
            else:
                self.phi = LogisticLoss().to(self.device)
                self.psi = LogisticLoss(alpha=1, c=self.c).to(self.device)
        else:
            self.params = list(self.predictor.parameters()) + \
                list(self.discri.parameters())

        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.predictor.parameters(), args.lr)

        self.step = 0

    def train_batch(self, X, y, epoch, args):
        """Trains the model for one batch.

        Args:
            X: Input images.
            y: Input labels for images.
            epoch: training epoch.
            args: Other arguments for training.
        """
        self.optimizer.zero_grad()

        X, y = X.to(self.device), y.to(self.device)
        y = F.one_hot(y).bool()

        if self.method == "Ours":
            # preds: (batch_size, num_classe)
            preds = self.predictor(X)
            rej_scores = self.rejector(X)

            loss = self.our_loss_fn(preds, rej_scores, y)

        elif self.method == "Ni+":
            # preds: (batch_size, num_classe)
            preds = self.predictor(X)
            rej_scores = self.rejector(X)

            loss = self.ni_et_al_loss_fn(preds, rej_scores, y)

        elif "HF" in self.method:
            pass

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def our_loss_fn(self, preds, rej_scores, y):
        preds_y = self.label_pred(preds, y)
        margin = self.prediction_margin(preds_y, preds, y)
        if self.our_surr_type == "MCS":
            loss = self.phi(margin) * self.psi_1(rej_scores).mean() \
                + self.psi_2(rej_scores).mean()

        elif self.our_surr_type == "ACS":
            loss = self.phi(margin - rej_scores).mean() \
                + self.psi_2(rej_scores).mean()
        else:
            raise Exception(
                "Method should be one of MCS and ACS")
        return loss

    def ni_et_al_loss_fn(self, preds, rej_scores, y):
        preds_y = self.label_pred(preds, y)

        diff = self.pw_diff(preds_y, preds, y)
        if self.ni_et_al_surr_type == "MCS":
            loss = self.phi(diff).sum(
                1) * self.psi_1(rej_scores).mean()\
                + self.psi_2(rej_scores).mean()

        elif self.ni_et_al_surr_type == "ACS":
            loss = self.phi(diff - rej_scores)\
                .sum(-1) * self.psi_1(rej_scores).mean()
            + self.psi_2(rej_scores).mean()

        elif self.ni_et_al_surr_type == "OVA":
            loss = self.phi(preds_y).mean() \
                + self.phi(self.pw_diff(0, preds, y)).sum(-1).mean()

        elif self.ni_et_al_surr_type == "CE":
            loss = -preds_y.mean()
            + torch.log(torch.exp(preds)).sum(-1).mean()
        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")
        return loss

    def test_batch(self, X, y):
        """Tests the model for one batch.

        Args:
            X: Input images.
            y: Input labels for images.
        """
        X, y = X.to(self.device), y.to(self.device)

        preds = self.predictor(X)

        if self.method == "Ours":
            whether_pred = self.our_decid_fn(X)
        elif self.method == "Ni+":
            whether_pred = self.ni_et_al_decid_fn(preds)
        preds, y = preds[whether_pred], y[whether_pred]

        _, pred_y = torch.max(preds.data, 1)

        n_correct = (pred_y == y).sum().item()

        return n_correct

    def our_decid_fn(self, X):
        return self.rejector(X) > 0

    def ni_et_al_decid_fn(self, preds):
        if self.ni_et_al_surr_type == "MCS" \
                or self.ni_et_al_surr_type == "ACS":
            whether_pred = torch.max(F.softmax(preds), -1)[0] > 1 - self.c

        elif self.ni_et_al_surr_type == "OVA":
            whether_pred = torch.max(F.sigmoid(preds), -1)[0] > 1 - self.c

        elif self.ni_et_al_surr_type == "CE":
            whether_pred = torch.max(F.softmax(preds), -1)[0] > 1 - self.c

        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")

        return whether_pred

    def save_params(self):
        ensure_dir(self.checkpoint_dir, verbose=True)
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     self.method + "_" + self.model_id + ".pt")
        params = self.predictor.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     self.method + "_" + self.model_id + ".pt")
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.predictor is not None:
            self.predictor.load_state_dict(checkpoint)
