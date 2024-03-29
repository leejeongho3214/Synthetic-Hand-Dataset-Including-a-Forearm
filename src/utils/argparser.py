import os
import argparse
from src.tools.models.our_net import get_our_net
import torch
import time
from tensorboardX import SummaryWriter
import numpy as np
from src.utils import *


def parse_args(eval=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        default="None",
        help="You write down to store the directory path",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="ours",
        type=str,
        help="Automatically detect dataset name by your name",
        required=False,
    )
    parser.add_argument("--count", default=4, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--graph_num", default=4, type=int)
    parser.add_argument(
        "--loss_aux", default=0, type=float, help="Multiple this value to hrnet loss"
    )
    parser.add_argument(
        "--loss_2d", default=0, help="Multiple this value to 2d loss", type=float
    )
    parser.add_argument(
        "--loss_3d", default=1, help="Multiple this value to 3d loss", type=float
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete the checkpoint folder"
    )
    parser.add_argument(
        "--which_gcn", default="0, 0, 1", help="Which encoder block you use", type=str
    )
    parser.add_argument("--backbone", default="hrnet", type=str)

    parser.add_argument(
        "--arm",
        action="store_true",
        help="Whether you crop the forearm in image",
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--ratio_of_dataset", default=1.0, type=float)
    parser.add_argument(
        "--num_hidden_layers",
        default=4,
        type=int,
        help="How many layer you use in a encoder blcok",
    )
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--center", action="store_true")
    args = parser.parse_args()
    args, logger = pre_arg(args, eval)
    args.logger = logger

    return args


def load_model(args):
    epoch = 0
    best_loss = np.inf
    count = 0
    args.num_gpus = 1
    args.device = torch.device(args.device)

    _model = get_our_net(args)
    log_dir = f"tensorboard/{args.name}"
    writer = SummaryWriter(log_dir)

    if args.reset:
        reset_folder(os.path.join(args.root_path, args.name))

    if os.path.isfile(
        os.path.join(args.root_path, args.name, "checkpoint-good/state_dict.bin")
    ):
        best_loss, epoch, _model, count = resume_checkpoint(
            _model,
            os.path.join(args.root_path, args.name, "checkpoint-good/state_dict.bin"),
        )
        args.logger.debug("Loading ===> %s" % os.path.join(args.root_path, args.name))
        print(
            colored(
                "Loading ===> %s" % os.path.join(args.root_path, args.name),
                "green",
            )
        )
        args.reset = "resume"
    else:
        args.reset = "init"

    _model.to(args.device)

    return _model, best_loss, epoch, count, writer


def train(
    args,
    train_dataloader,
    test_dataloader,
    Graphormer_model,
    epoch,
    best_loss,
    data_len,
    count,
    writer,
    len_total,
    batch_time,
):
    end = time.time()
    phase = "TRAIN"
    runner = Runner(
        args,
        Graphormer_model,
        epoch,
        train_dataloader,
        test_dataloader,
        phase,
        batch_time,
        data_len,
        len_total,
        count,
        best_loss,
        writer,
    )

    Graphormer_model, optimizer, batch_time = runner.our(end)

    return Graphormer_model, optimizer, batch_time, best_loss


def valid(
    args,
    train_dataloader,
    test_dataloader,
    Graphormer_model,
    epoch,
    count,
    best_loss,
    data_len,
    writer,
    batch_time,
    len_total,
):
    end = time.time()
    phase = "VALID"
    runner = Runner(
        args,
        Graphormer_model,
        epoch,
        train_dataloader,
        test_dataloader,
        phase,
        batch_time,
        data_len,
        len_total,
        count,
        best_loss,
        writer,
    )

    loss, count, batch_time = runner.our(end)

    return loss, count,  batch_time
