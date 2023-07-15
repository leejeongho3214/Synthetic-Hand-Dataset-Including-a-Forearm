import gc
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import torch
from torch.utils import data
from src.utils.argparser import parse_args, load_model, train, valid
from dataset import *
from src.utils.bar import colored

def worker(world_rank, world_size, nodes_size, args, trainset_loader, valset_loader, train_dataset, val_dataset):

    _model, best_loss, epo, count, writer = load_model(args)
    pck_l = 0; batch_time = AverageMeter()
    print(colored("Train_len: {}, Val_len: {}".format(len(train_dataset), len(val_dataset)), "blue"))
    
    args.logger.debug("Train_len: {}, Val_len: {}".format(len(train_dataset), len(val_dataset)))
    if world_rank >= 0:
        assert torch.cuda.is_available() and torch.cuda.device_count() > world_rank
        torch.cuda.empty_cache()
        args.device = torch.device("cuda", world_rank)
        
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:23456" if nodes_size == 1 else "env://",
            rank=world_rank, world_size=world_size)
        torch.cuda.set_device(args.device)
    
    for epoch in range(epo, args.epoch):
        Graphormer_model, optimizer, batch_time, best_loss = train(args, trainset_loader, valset_loader, _model, epoch, best_loss, len(train_dataset), count, writer, pck_l, len(trainset_loader)+len(valset_loader), batch_time)
        loss, count, pck, batch_time = valid(args, trainset_loader, valset_loader, Graphormer_model, epoch, count, best_loss, len(train_dataset),  writer, batch_time, len(trainset_loader)+len(valset_loader), pck_l)
        pck_l = max(pck, pck_l)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        if is_best:
            count = 0
            _model = Graphormer_model
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, count,  'good',logger= args.logger)
            del Graphormer_model

        else:
            count += 1
            if count == args.count:
                break
        gc.collect()
        torch.cuda.empty_cache()

def main(args):
        
    train_dataset, val_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valset_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    
    device_ids = list(map(int, args.device_ids.split(",")))
    nprocs = len(device_ids)
    
    if nprocs > 1:
        torch.multiprocessing.spawn(
            worker, args=(nprocs, 1,  args, trainset_loader, valset_loader, train_dataset, val_dataset), nprocs=nprocs,
            join=True)
    elif nprocs == 1:
        worker(device_ids[0], nprocs, 1, args, trainset_loader, valset_loader, train_dataset, val_dataset )
    else:
        assert False
    
    

  

if __name__ == "__main__":
    args= parse_args()
    main(args)
