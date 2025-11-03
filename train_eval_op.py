# -*- coding: utf-8 -*-
import gc
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.train_utils import EarlyStopping, LRDecay
from trainer import ModelTrainer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_op(trainer, train_dataloader, epoch, args):
    """Trains model for one epoch.

    Args:
        trainer: Trainer.
        train_data_loader: Dataloader to load the train data. 
        epoch: Training epoch.
        args: Other arguments for training.
    """
    trainer.model.train()

    loss = 0
    step = 0
    for X, y in train_dataloader:
        batch_loss = trainer.train_batch(
            X, y, epoch, args)
        loss += batch_loss
        step += 1

        gc.collect()
    logging.info("Epoch {}/{} - Training Loss: {:.3f}".format(
        epoch, args.num_epoch, loss / step))


def cal_test_score(n_correct, n_samples):
    return n_correct / n_samples


def evaluation_logging(eval_log, epoch, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % epoch)
    else:
        logging.info("Test:")

    logging.info("ACC: %.4f" % eval_log["ACC"])

    return eval_log


def eval_op(trainer, dataloader, epoch, mode):
    """Evaluates one client with its own valid/test data for one epoch.

    Args:
        mode: `valid` or `test`.
    """
    trainer.model.eval()

    n_samples, n_correct = 0, 0
    for X, y in dataloader:
        batch_n_correct = trainer.test_batch(
            X, y)
        n_correct += batch_n_correct
        n_samples += y.shape[0]

    gc.collect()
    ACC = cal_test_score(n_correct, n_samples)
    eval_log = {"ACC": ACC}

    evaluation_logging(
        eval_log, epoch, mode)

    return eval_log
