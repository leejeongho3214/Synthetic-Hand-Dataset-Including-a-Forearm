import gc
import shutil
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
<<<<<<< HEAD
=======

>>>>>>> 41be42ec447305464f23a500fdf08eca23119004
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
from argparser import parse_args, load_model, train, test
from dataset import *

def main(args):

    _model, logger, best_loss, epo, count = load_model(args)
    train_dataset, test_dataset = build_dataset(args)

    logger.info('\n \n========================================================================================================\n' 
                + 'name = %s, model = %s,  epoch = %i, count = %i, dataset = %s, Train images = %i, \n 2d = %i , 3d = %i,  ratio of aug = %.1f, color = %s, general = %s, memo = %s'
                + '\n========================================================================================================',
                 args.name, args.model, args.epoch, args.count, args.dataset, len(train_dataset), args.loss_2d, args.loss_3d, args.ratio_of_aug, args.color, args.general, args.memo)

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    log_dir = f'tensorboard/{args.output_dir}'
    writer = SummaryWriter(log_dir); pck_l = 0; batch_time = AverageMeter()

    
    for epoch in range(epo, args.epoch):
        Graphormer_model, optimizer, batch_time = train(args, trainset_loader, _model, epoch, best_loss, len(train_dataset),logger, count, writer, pck_l, len(trainset_loader)+len(testset_loader), batch_time)
        loss, count, pck, batch_time = test(args, testset_loader, Graphormer_model, epoch, count, best_loss, logger, writer, batch_time, len(trainset_loader)+len(testset_loader))
        
        pck_l = max(pck, pck_l)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        if is_best:
            count = 0
            _model = Graphormer_model
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, count,  'good',logger=logger)
            del Graphormer_model

        else:
            count += 1
            if count == args.count:
                break

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    main(args)
