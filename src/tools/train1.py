import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils import data
from src.utils.argparser import parse_args, load_model, train, valid
from dataset import *
from src.utils.bar import colored


def main(args):
    train_dataset, val_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    _model, best_loss, epo, count, writer = load_model(args)
    batch_time = AverageMeter()
    print(
        colored(
            "Train_len: {}, Val_len: {}".format(len(train_dataset), len(val_dataset)),
            "blue",
        )
    )
    args.logger.debug(
        "Train_len: {}, Val_len: {}".format(len(train_dataset), len(val_dataset))
    )
    best_loss = np.inf
    for epoch in range(epo, args.epoch):
        Graphormer_model, optimizer, batch_time, best_loss = train(
            args,
            trainset_loader,
            valset_loader,
            _model,
            epoch,
            best_loss,
            len(train_dataset),
            count,
            writer,
            len(trainset_loader) + len(valset_loader),
            batch_time,
        )
        loss, count, batch_time = valid(
            args,
            trainset_loader,
            valset_loader,
            Graphormer_model,
            epoch,
            count,
            best_loss,
            len(train_dataset),
            writer,
            batch_time,
            len(trainset_loader) + len(valset_loader),
        )
        
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            count = 0
            _model = Graphormer_model
            save_checkpoint(
                Graphormer_model,
                args,
                epoch,
                optimizer,
                best_loss,
                count,
                "good",
                logger=args.logger,
            )
            del Graphormer_model

        else:
            count += 1
            if count == args.count:
                break

if __name__ == "__main__":
    args = parse_args()
    main(args)
