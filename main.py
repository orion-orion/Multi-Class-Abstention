# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import random
import argparse
import torch
from trainer import ModelTrainer
import logging
from client import Client
from server import Server
from utils.data_utils import load_dataset
from utils.io_utils import save_config, ensure_dir
from experiment import run_experiment


def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset part
    parser.add_argument(dest="--dataset", metavar="--dataset", nargs="*",
                        help="name of dataset;"
                        " possible are `SVHN`, `CIFAR10`，`CIFAR100`")

    # Training part
    parser.add_argument("--method", type=str, default="Ours",
                        help="method, possible are `Ours`, `Ni+`, "
                        "`LocalHF`, `FedDHCF`, `LocalDHCF`, `FedMF`, "
                        "`LocalMF`, `FedGNN`, `LocalGNN`, `LocalP2FCDR`, "
                        "`FedP2FCDR`, `FedPPDM`, `LocalPPDM`")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--num_epoch", type=int, default=40,
                        help="Number of total training epochs.")
    parser.add_argument("--local_epoch", type=int, default=3,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", choices=["sgd", "adagrad", "adam",
                                                "adamax"], default="adam",
                        help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to sgd and adagrad.")  # 0.001
    parser.add_argument("--lr_decay", type=float, default=0.9,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=10,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--our_surr_type", type=str, default="MCS",
                        help="Type of our surrogate losses, MCS or ACS.")
    parser.add_argument("--ni_et_al_surr_type", type=str, default="MCS",
                        help="Type of Ni et al.'s surrogate loss, MCS, ACS,"
                        "OVA or CE.")
    parser.add_argument("--psi_type", type=str, default="exponential",
                        help="Type of function psi, exponential or hinge.")
    parser.add_argument("--c", type=float, default=0.1,
                        help="Abstention cost.")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Hyperparameter to control the performance of \
                            the rejector.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--model_id", type=str, default=str(int(time.time())),
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")

    # Hypergraph signal decoupling arguments for FedHCDR method (ours)
    # 2.0 for FKCB, 3.0 for SCEC, and 1.0 for SGHT is the best
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Coefficient of local-global bi-directional "
                        "knowledge transfer loss (MI) for FedHCDR method "
                        "(ours). 2.0 for FKCB, 3.0 for SCEC, and 1.0 for SGHT "
                        "is the best")
    parser.add_argument("--n_rw", type=int, default=128,
                        help="Steps of hypergraph random walk for FedHCDR "
                        "method (ours). 128 is the best")
    parser.add_argument("--wait_epoch", type=int, default=0,
                        help="epochs to wait before performing local-global "
                        "bi-directional knowledge transfer")

    # Contrastive arguments for FedHCDR method (ours)
    # 2.0 for FKCB, 1.0 for SCEC, and 3.0 for SGHT is the best
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Coefficient of hypergraph contrastive loss for "
                        "FedHCDR method (ours). 2.0 for FKCB, 1.0 for SCEC, "
                        "and 3.0 for SGHT is the best")
    parser.add_argument("--drop_edge_rate", type=float, default=0.05,
                        help="Rate of edges being dropped in graph "
                        "perturbation for FedHCDR method (ours). 0.05 is the "
                        "best")

    args = parser.parse_args()
    assert (args.method in ["FedHCDR", "FedHF", "LocalHF",
                            "FedDHCF", "LocalDHCF", "FedMF",
                            "LocalMF", "FedGNN", "LocalGNN",
                            "FedPriCDR", "LocalPriCDR",
                            "FedP2FCDR", "LocalP2FCDR",
                            "FedPPDM", "LocalPPDM"])
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def init_logger(args):
    """Init a file logger that opens the file periodically and write to it.
    """
    log_path = os.path.join(args.log_dir,
                            "domain_" + "".join([domain[0] for domain
                                                 in args.domains]))
    ensure_dir(log_path, verbose=True)

    model_id = args.model_id if len(args.model_id) > 1 else "0" + args.model_id
    log_file = os.path.join(log_path, args.method + "_" + model_id + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()

    seed_everything(args)

    init_logger(args)

    train_dataset, valid_dataset, test_dataset, data_info = load_dataset(args)

    # Save the config of input arguments
    save_config(args)

    run_experiment(train_dataset, valid_dataset, test_dataset, data_info, args)


if __name__ == "__main__":
    main()
