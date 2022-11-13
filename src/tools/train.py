import gc
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from torch.utils import data
from torch.utils.data import random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from argparser import parse_args, load_model, train, test
from dataset import *
from src.datasets.build import make_hand_data_loader
from datetime import datetime
# sys.path.append("C:\\Users\\jeongho\\PycharmProjects\\PoseEstimation\\HandPose\\MeshGraphormer-main")

def main(args):

    
    _model, logger, best_loss, epo, count = load_model(args)
    path = "../../datasets/1102"
    general_path = "../../datasets/1108"

    folder_num = os.listdir(path)

    for iter, degree in enumerate(folder_num):
        
        dataset = CustomDataset_train_new(degree, path, rotation = args.rot, color = args.color, blur = args.blur, erase = args.erase, ratio = args.ratio_of_aug, ratio2 = 1)
    
        if iter == 0:
            train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

        else:
            train_dataset_new, test_dataset_new = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
            train_dataset  = ConcatDataset([train_dataset, train_dataset_new])        
            test_dataset = ConcatDataset([test_dataset, test_dataset_new])    
    
    if args.frei:
        _, _ , train_dataset1, test_dataset1 = make_hand_data_loader(args, args.train_yaml,
                                                                            args.distributed, is_train=True,
                                                                            scale_factor=args.img_scale_factor) ## RGB image
        train_dataset = ConcatDataset([train_dataset1, train_dataset])
        test_dataset = ConcatDataset([test_dataset1, test_dataset])

    if args.general:

        folder_num = os.listdir(general_path)
        for iter, degree in enumerate(folder_num):
            
            dataset = CustomDataset_train_new(degree, general_path, rotation = args.rot, color = args.color, blur = args.blur, erase = args.erase, ratio = args.ratio_of_aug, ratio2 = 0.137)
        
            if iter == 0:
                train_dataset1, test_dataset1 = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

            else:
                train_dataset_new, test_dataset_new = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                train_dataset1  = ConcatDataset([train_dataset1, train_dataset_new])        
                test_dataset1 = ConcatDataset([test_dataset1, test_dataset_new])    
                
        train_dataset = ConcatDataset([train_dataset1, train_dataset])
        test_dataset = ConcatDataset([test_dataset1, test_dataset])
        
    param = f"{args.output_dir}/frei_{args.frei}_rot_{args.rot}_color_{args.color}_blur_{args.blur}_erase_{args.erase}_ratio_{args.ratio_of_aug}_dataset_{len(train_dataset)}"
    mkdir(param)
    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    logger.info('%s', args)
    writer = SummaryWriter(log_dir = f'tensorboard/{param}')
    pck_l = 0
    batch_time = AverageMeter()
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
