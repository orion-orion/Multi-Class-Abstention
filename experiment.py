# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.train_utils import EarlyStopping, LRDecay
from trainer import ModelTrainer
from train_eval_op import train_op, eval_op


def load_and_eval_model(trainer, test_dataloader):
    trainer.load_params()
    eval_op(trainer, test_dataloader, 0, mode="test")


def training_loop(trainer, train_dataloader, valid_dataloader, test_dataloader,
                  args, mode="regular"):
    trainer.mode = mode
    whether_valid = True if valid_dataloader else False
    trainer.init_optimzer(args, mode, whether_valid)
    early_stopping = EarlyStopping(
        args.checkpoint_dir, patience=args.es_patience, verbose=True)
    lr_decay = LRDecay(args.lr, args.decay_epoch,
                       args.optimizer, args.lr_decay,
                       patience=args.ld_patience, verbose=True)

    for epoch in tqdm(range(1, args.num_epoch + 1), ascii=True):
        # Train one epoch
        train_op(trainer, train_dataloader, epoch, args)

        if whether_valid and epoch % args.eval_interval == 0:
            eval_log = eval_op(trainer, valid_dataloader,
                               epoch, mode="valid")

            # Early Stopping. Here only compare the current results with
            # the best results
            early_stopping(eval_log["ACC_all"], trainer)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            # Learning rate decay. Here only compare the current results
            # with the latest results
            lr_decay(epoch, eval_log["ACC_all"], trainer)
            # trainer.scheduler.step(epoch)
        else:
            eval_log = eval_op(trainer, test_dataloader,
                               epoch, mode="test")


def run_experiment(train_dataset, valid_dataset, test_dataset, data_info, args):
    trainer = ModelTrainer(args, data_info)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    if valid_dataset:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        valid_dataloader = None
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.do_eval:
        load_and_eval_model(trainer, test_dataloader)
    else:
        if (args.method == "Mao+" and args.mao_mode == "two-stage") \
                or (args.method == "Ours" and args.ours_mode == "two-stage"):
            logging.info("****************First Stage****************")
            training_loop(trainer, train_dataloader,
                          valid_dataloader, test_dataloader, args, mode="stage1")
            logging.info("****************Second Stage****************")
            training_loop(trainer, train_dataloader,
                          valid_dataloader, test_dataloader, args, mode="stage2")
        else:
            training_loop(trainer, train_dataloader,
                          valid_dataloader, test_dataloader, args)

        load_and_eval_model(trainer, test_dataloader)
