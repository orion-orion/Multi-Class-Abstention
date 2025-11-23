# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.cnn.resnet import resnet20, resnet32, resnet110
from utils.io_utils import ensure_dir
from utils import train_utils
from losses import OurLoss, MaoLoss, NiLoss
import logging


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, data_info):
        self.args = args
        self.dataset = args.dataset
        self.method = args.method
        self.mode = "regular"
        if args.cuda:
            torch.cuda.empty_cache()
            self.device = "cuda:%s" % args.gpu
        else:
            self.device = "cpu"
        # self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.predictor = resnet32(in_channels=data_info["n_channels"],
                                  output_size=data_info["num_cls"])\
            .to(self.device)
        self.rejector = resnet32(in_channels=data_info["n_channels"],
                                 output_size=1).to(self.device)
        self.c = args.c
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = (args.model_id if len(args.model_id)
                         > 1 else "0" + args.model_id)
        if self.method == "Ours":
            self.loss_fn = OurLoss(
                args.our_surr_type, args.psi_type, alpha=args.alpha, c=args.c)\
                .to(self.device)

        elif self.method == "Ni+":
            self.loss_fn = NiLoss(
                args.ni_surr_type, alpha=args.alpha, c=args.c).to(self.device)
            self.ni_surr_type = args.ni_surr_type

        elif self.method == "Mao+":
            self.loss_fn = MaoLoss(
                args.mao_l_type, alpha=args.alpha, c=args.c).to(self.device)

        else:
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def init_optimzer(self, args, mode="regular", whether_valid=True):
        if mode == "regular":
            if self.method in ["Ours", "Ni+", "Mao+"]:
                self.params = list(self.predictor.parameters()) + \
                    list(self.rejector.parameters())
            else:
                self.params = self.predictor.parameters()

        elif mode == "stage1":
            self.params = self.predictor.parameters()
        else:
            self.params = self.rejector.parameters()

        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.lr, args.weight_decay)
        self.whether_valid = whether_valid
        if not self.whether_valid:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=(4000, 6000), gamma=0.1)
            # T_0 = 10
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     self.optimizer, T_0)
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
        if self.method in ["Ours", "Ni+", "Mao+"] and self.mode == "regular":
            y = F.one_hot(y).bool()

        # preds: (batch_size, num_classe)
        preds = self.predictor(X)
        rej_scores = self.rejector(X)

        if self.method == "Mao+":
            if self.mode == "stage1":
                loss = self.loss_fn.forward_stage1(preds, y)
            elif self.mode == "stage2":
                loss = self.loss_fn.forward_stage2(preds, rej_scores, y)
            else:
                loss = self.loss_fn(preds, rej_scores, y)
        elif self.method == "Ours":
            if self.mode == "stage1":
                loss = self.loss_fn.forward_stage1(preds, y)
            elif self.mode == "stage2":
                loss = self.loss_fn.forward_stage2(preds, rej_scores, y)
            else:
                loss = self.loss_fn(preds, rej_scores, y)

        elif self.method in ["Ours", "Ni+"]:
            loss = self.loss_fn(preds, rej_scores, y)
        else:
            loss = self.loss_fn(preds, y)

        loss.backward()
        self.optimizer.step()
        if not self.whether_valid:
            self.scheduler.step()
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

        if self.method == "Ours":
            if self.mode == "stage1":
                whether_pred = torch.ones(preds.shape[0]).bool()
            else:
                whether_pred = self.rej_decid_fn(X)
        elif self.method == "Mao+":
            if self.mode == "stage1":
                whether_pred = torch.ones(preds.shape[0]).bool()
            else:
                whether_pred = self.rej_decid_fn(X)
        elif self.method == "Ni+":
            whether_pred = self.conf_decid_fn(preds)
        else:
            whether_pred = torch.ones(preds.shape[0]).bool()

        preds, y = preds[whether_pred], y[whether_pred]

        _, pred_y = torch.max(preds.data, 1)

        n_correct = (pred_y == y).sum().item()
        n_accept = whether_pred.sum().item()
        n_reject = whether_pred.shape[0] - n_accept
        n_error = n_accept - n_correct
        abst_loss = self.abstention_loss(n_error, n_reject, self.c)

        return n_correct, n_error, n_accept, n_reject, abst_loss

    def rej_decid_fn(self, X):
        # logging.info("predictor's output:", self.predictor(X))
        # logging.info("rejector's output:", self.rejector(X))
        return (self.rejector(X) > 0).view(-1)

    def conf_decid_fn(self, preds):
        if self.ni_surr_type == "MCS" or self.ni_surr_type == "ACS":
            whether_pred = torch.max(
                F.softmax(preds, dim=-1), -1)[0] > 1 - self.c

        elif self.ni_surr_type == "OVA":
            whether_pred = torch.max(F.sigmoid(preds), -1)[0] > 1 - self.c

        elif self.ni_surr_type == "CE":
            whether_pred = torch.max(
                F.softmax(preds, dim=-1), -1)[0] > 1 - self.c

        else:
            raise Exception(
                "Surrogate should be one of MCS, ACS, OVA, and CE")

        return whether_pred.view(-1)

    @staticmethod
    def abstention_loss(n_error, n_reject, c):
        return n_error + c * n_reject

    def save_params(self):
        ensure_dir(self.checkpoint_dir, verbose=True)
        ckpt_filename = os.path.join(self.checkpoint_dir, self.dataset,
                                     self.method + "_" + self.model_id + ".pt")
        params = self.predictor.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir, self.dataset,
                                     self.method + "_" + self.model_id + ".pt")
        try:
            checkpoint = torch.load(ckpt_filename, weights_only=True)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.predictor is not None:
            self.predictor.load_state_dict(checkpoint)
