# -*- coding: utf-8 -*-
import gc
import logging
import torch


def train_op(trainer, train_dataloader, epoch, args):
    """Trains model for one epoch.

    Args:
        trainer: Trainer.
        train_data_loader: Dataloader to load the train data.
        epoch: Training epoch.
        args: Other arguments for training.
    """
    trainer.predictor.train()
    if not args.method == "CE" \
            and not (args.method == "Ni+" and args.ni_surr_type
                     in ["OVA", "CE"]) and not trainer.mode == "stage1":
        if not args.method == "Ours":
            trainer.rejector.train()

    loss = 0
    n_samples, n_correct, n_accept = 0, 0, 0
    for X, y in train_dataloader:
        batch_loss, batch_correct, batch_accept = trainer.train_batch(
            X, y, epoch, args)
        loss += batch_loss
        n_correct += batch_correct
        n_accept += batch_accept
        n_samples += y.shape[0]

        gc.collect()
    logging.info("Epoch {}/{} - Training Loss: {:.3f}, Training ACC: {:.4f}"
                 .format(epoch, args.num_epoch, loss / n_samples,
                         n_correct / (n_accept + 1e-6)))


def cal_test_score(surr_loss, n_correct, n_error, n_accept, n_reject,
                   abst_loss, n_samples):
    surr_loss = surr_loss / n_samples
    ACC_acpt = n_correct / (n_accept + 1e-6)
    ACC_all = n_correct / n_samples
    abst_loss = abst_loss / n_samples
    misclassf_error = n_error / (n_accept + 1e-6)
    rej_ratio = n_reject / n_samples

    return surr_loss, ACC_acpt, ACC_all, abst_loss, misclassf_error, rej_ratio


def evaluation_logging(eval_log, epoch, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % epoch)
    else:
        logging.info("Test:")

    logging.info("Surrogate loss: %.4f \t Abstention loss: %.4f \t "
                 "Rejection ratio: %.4f" % (eval_log["Surrogate loss"],
                                            eval_log["Abstention loss"],
                                            eval_log["Rejection ratio"]))
    logging.info("ACC_acpt: %.4f \t ACC_all: %.4f \t "
                 "Misclassification err: %.4f" %
                 (eval_log["ACC_acpt"],
                  eval_log["ACC_all"],
                  eval_log["Misclassification err"]))

    return eval_log


def eval_op(trainer, dataloader, epoch, args, mode):
    """Evaluates one client with its own valid/test data for one epoch.

    Args:
        mode: `valid` or `test`.
    """
    trainer.predictor.eval()
    if not args.method == "CE" \
            and not (args.method == "Ni+" and args.ni_surr_type
                     in ["OVA", "CE"]) and not trainer.mode == "stage1":
        if not args.method == "Ours":
            trainer.rejector.eval()

    surr_loss, abst_loss = 0.0, 0.0
    n_samples,  n_correct, n_error, n_accept, n_reject = 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            batch_surr_loss, batch_n_correct, batch_n_error, batch_n_accept, \
                batch_n_reject, batch_abst_loss = trainer.test_batch(X, y)

            surr_loss += batch_surr_loss
            n_correct += batch_n_correct
            n_error += batch_n_error
            n_accept += batch_n_accept
            n_reject += batch_n_reject
            abst_loss += batch_abst_loss
            n_samples += y.shape[0]

    gc.collect()
    surr_loss, ACC_acpt, ACC_all, abst_loss, misclassf_error, rej_ratio \
        = cal_test_score(surr_loss, n_correct, n_error, n_accept, n_reject,
                         abst_loss, n_samples)
    eval_log = {"Surrogate loss": surr_loss, "ACC_acpt": ACC_acpt,
                "ACC_all": ACC_all, "Abstention loss": abst_loss,
                "Misclassification err": misclassf_error,
                "Rejection ratio": rej_ratio}

    evaluation_logging(
        eval_log, epoch, mode)

    return eval_log
