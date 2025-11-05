# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.resnet import resnet110
from utils.io_utils import ensure_dir
from utils import train_utils
from losses import OurLoss, MaoLoss, NiLoss


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
        if args.cuda:
            torch.cuda.empty_cache()
            self.device = "cuda:%s" % args.gpu
        else:
            self.device = "cpu"
        # self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.predictor = resnet110(in_channels=data_info["n_channels"],
                                   output_size=data_info["num_cls"])\
            .to(self.device)
        self.rejector = resnet110(in_channels=data_info["n_channels"],
                                  output_size=1).to(self.device)
        self.c = args.c
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = (args.model_id if len(args.model_id)
                         > 1 else "0" + args.model_id)
        self.bce_criterion = nn.BCEWithLogitsLoss().to(self.device)
        if self.method == "Ours":
            self.loss_fn = OurLoss(
                args.our_surr_type, args.psi_type, alpha=1, c=1)\
                .to(self.device)

        elif self.method == "Ni+":
            self.loss_fn = NiLoss(args.ni_surr_type).to(self.device)

        elif self.method == "Mao+":
            self.loss_fn = MaoLoss(args.mao_l_type).to(self.device)
        else:
            pass

        self.params = list(self.predictor.parameters()) + \
            list(self.rejector.parameters())

        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.lr)

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

        # preds: (batch_size, num_classe)
        preds = self.predictor(X)
        rej_scores = self.rejector(X)

        loss = self.loss_fn(preds, rej_scores, y)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def test_batch(self, X, y):
        """Tests the model for one batch.

        Args:
            X: Input images.
            y: Input labels for images.
        """
        X, y = X.to(self.device), y.to(self.device)

        preds = self.predictor(X)

        if self.method == "Ours" or self.method == "Mao+":
            whether_pred = self.rej_decid_fn(X)
        elif self.method == "Ni+":
            whether_pred = self.conf_decid_fn(preds)
        preds, y = preds[whether_pred], y[whether_pred]

        _, pred_y = torch.max(preds.data, 1)

        n_correct = (pred_y == y).sum().item()

        return n_correct

    def rej_decid_fn(self, X):
        return (self.rejector(X) > 0).view(-1)

    def conf_decid_fn(self, preds):
        if self.ni_surr_type == "MCS" or self.ni_surr_type == "ACS":
            whether_pred = torch.max(F.softmax(preds), -1)[0] > 1 - self.c

        elif self.ni_surr_type == "OVA":
            whether_pred = torch.max(F.sigmoid(preds), -1)[0] > 1 - self.c

        elif self.ni_surr_type == "CE":
            whether_pred = torch.max(F.softmax(preds), -1)[0] > 1 - self.c

        else:
            raise Exception(
                "Method should be one of MCS, ACS, OVA, and CE")

        return whether_pred.view(-1)

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
