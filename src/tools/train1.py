import gc
import shutil
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model") ## Set your local file location for import library
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
from argparser import parse_args, load_model, train, test
from dataset import *

def main(args):

    _model, logger, best_loss, epo, count = load_model(args)
    train_dataset, test_dataset = build_dataset(args)

<<<<<<< HEAD
    param = f"model_{args.model}_general_{args.general}_2d_{args.loss_2d}_3d_{args.loss_3d}_rot_{args.rot}_color_{args.color}_ratio_{args.ratio_of_aug}_dataset_{len(train_dataset)}"
    for name in os.listdir(os.path.join(args.root_path, args.name)): 
        if not name in ["checkpoint-good", "train_image", "test_image", "log.txt"]: os.rmdir(os.path.join(f"output/{args.name}", name))## easy to check hyper-parameter by making empty folder
    mkdir(os.path.join(os.path.join(args.root_path, args.name),param))
    logger.info('\n \n========================================================================================================\n' 
                + 'name = %s, model = %s,  epoch = %i, count = %i, dataset = %s, Train images = %i, \n 2d = %i , 3d = %i,  ratio of aug = %.1f, rot= %s, color = %s, Use general = %s' 
                + '\n========================================================================================================',
                 args.name, args.model, args.epoch, args.count, args.dataset, len(train_dataset), args.loss_2d, args.loss_3d, args.ratio_of_aug, args.rot, args.color, args.general)
=======
    logger.info('\n \n========================================================================================================\n' 
                + 'name = %s, model = %s,  epoch = %i, count = %i, dataset = %s, Train images = %i, \n 2d = %i , 3d = %i,  ratio of aug = %.1f, color = %s, general = %s, memo = %s'
                + '\n========================================================================================================',
                 args.name, args.model, args.epoch, args.count, args.dataset, len(train_dataset), args.loss_2d, args.loss_3d, args.ratio_of_aug, args.color, args.general, args.memo)
>>>>>>> 75095027a1e1ec114691fcfe220c154ef0b276bb

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    log_dir = f'tensorboard/{args.output_dir}'
<<<<<<< HEAD
    if os.path.isdir(log_dir): shutil.rmtree(log_dir); mkdir(log_dir); mkdir(os.path.join(log_dir, param))
    else: mkdir(log_dir); mkdir(os.path.join(log_dir, param))
    writer = SummaryWriter(log_dir = os.path.join(log_dir, param)); pck_l = 0; batch_time = AverageMeter()
=======
    writer = SummaryWriter(log_dir); pck_l = 0; batch_time = AverageMeter()
>>>>>>> 75095027a1e1ec114691fcfe220c154ef0b276bb

    
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
