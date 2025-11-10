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


def run_experiment(train_dataset, valid_dataset, test_dataset, data_info, args):
    trainer = ModelTrainer(args, data_info)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.do_eval:
        load_and_eval_model(trainer, test_dataloader)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_dir, patience=args.es_patience, verbose=True)
        lr_decay = LRDecay(args.lr, args.decay_epoch,
                           args.optimizer, args.lr_decay,
                           patience=args.ld_patience, verbose=True)

        for epoch in tqdm(range(1, args.num_epoch + 1), ascii=True):
            # Train one epoch
            train_op(trainer, train_dataloader, epoch, args)

            if epoch % args.eval_interval == 0:
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
                lr_decay(epoch, eval_log, trainer)

        load_and_eval_model(trainer, test_dataloader)
