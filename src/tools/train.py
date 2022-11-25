import gc
import shutil
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model") ## Set your local file location for import library
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
from argparser import parse_args, load_model, train, test
from dataset import *

def main(args):

    _model, logger, best_loss, epo, count = load_model(args)
    train_dataset, test_dataset = build_dataset(args)

    param = f"model_{args.model}_general_{args.general}_frei_{args.frei}_rot_{args.rot}_color_{args.color}_blur_{args.blur}_erase_{args.erase}_ratio_{args.ratio_of_aug}_dataset_{len(train_dataset)}"
    for name in os.listdir(os.path.join(args.root_path, args.output_path)): 
        if not name in ["checkpoint-good", "train_image", "test_image", "log.txt"]: os.rmdir(os.path.join(f"output/{args.output_path}", name)) ## easy to check hyper-parameter by making empty folder
    logger.info('%s', args)

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    log_dir = f'tensorboard/{args.output_dir}'
    if os.listdir(log_dir): shutil.rmtree(log_dir); mkdir(log_dir); mkdir(os.path.join(log_dir, param))
    writer = SummaryWriter(log_dir = os.path.join(log_dir, param)); pck_l = 0; batch_time = AverageMeter()

    
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
