# -*- coding: utf-8 -*-
import gc
import logging


def train_op(trainer, train_dataloader, epoch, args):
    """Trains model for one epoch.

    Args:
        trainer: Trainer.
        train_data_loader: Dataloader to load the train data. 
        epoch: Training epoch.
        args: Other arguments for training.
    """
    trainer.predictor.train()
    trainer.rejector.train()

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


def cal_test_score(n_correct, n_error, n_accept, n_reject,
                   abst_loss, n_samples):
    acc = n_correct / (n_accept + 1e-6)
    acc_all = n_correct / n_samples
    abst_loss = abst_loss / n_samples
    misclassf_error = n_error / (n_accept + 1e-6)
    rej_ratio = n_reject / n_samples

    return acc, acc_all, abst_loss, misclassf_error, rej_ratio


def evaluation_logging(eval_log, epoch, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % epoch)
    else:
        logging.info("Test:")

    logging.info("ACC: %.4f \t ACC_All: %.4f \t Abstention loss: %.4f"
                 % (eval_log["ACC"], eval_log["ACC_All"],
                     eval_log["Abstention loss"]))
    logging.info("Misclassification err: %.4f \t Rejection ratio: %.4f"
                 % (eval_log["Misclassification err"],
                     eval_log["Rejection ratio"]))

    return eval_log


def eval_op(trainer, dataloader, epoch, mode):
    """Evaluates one client with its own valid/test data for one epoch.

    Args:
        mode: `valid` or `test`.
    """
    trainer.predictor.eval()
    trainer.rejector.eval()

    abst_loss = 0.0
    n_samples,  n_correct, \
        n_error, n_accept, n_reject = 0, 0, 0, 0, 0

    for X, y in dataloader:
        batch_n_correct, batch_n_error, batch_n_accept, \
            batch_n_reject, batch_abst_loss = trainer.test_batch(X, y)
        n_correct += batch_n_correct
        n_error += batch_n_error
        n_accept += batch_n_accept
        n_reject += batch_n_reject
        abst_loss += batch_abst_loss
        n_samples += y.shape[0]

    gc.collect()
    acc, acc_all, abst_loss, misclassf_error, rej_ratio = cal_test_score(
        n_correct, n_error, n_accept, n_reject, abst_loss, n_samples)
    eval_log = {"ACC": acc, "ACC_All": acc_all, "Abstention loss": abst_loss,
                "Misclassification err": misclassf_error,
                "Rejection ratio": rej_ratio}

    evaluation_logging(
        eval_log, epoch, mode)

    return eval_log
