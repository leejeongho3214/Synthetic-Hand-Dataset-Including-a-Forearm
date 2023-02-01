import gc
import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from torch.utils import data
from src.utils.argparser import parse_args, load_model, train, valid
from dataset import *
from src.utils.bar import colored
# from pytorch3d.docs.tutorials.utils import image_grid

def main(args):

    train_dataset, test_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    _model, best_loss, epo, count, writer = load_model(args)
    pck_l = 0; batch_time = AverageMeter()
    for epoch in range(epo, args.epoch):
        if epoch == epo: 
            print(colored(f"Path: {args.output_dir} | Dataset: {args.dataset} | Train_len: {len(train_dataset)} | Test_len: {len(test_dataset)} | Model: {args.model} | Status: {args.reset} | add_our: {(args.ratio_of_our * 100):.0f}% | align: {args.set} | 2D_loss: {args.loss_2d} | 3D_loss: {args.loss_3d} | 3D_mid_loss: {args.loss_3d_mid} | color: {(args.ratio_of_aug * 100):.0f}% | Max_count : {args.count} | Max_epoch : {args.epoch}", color = "yellow"))
            args.logger.debug("==" * 150)
            args.logger.debug(f"Path: {args.output_dir} | Dataset: {args.dataset} | Train_len: {len(train_dataset)} | Test_len: {len(test_dataset)} | Model: {args.model} | Status: {args.reset} | add_our: {(args.ratio_of_our * 100):.0f}% | align: {args.set} | 2D_loss: {args.loss_2d} | 3D_loss: {args.loss_3d} | 3D_mid_loss: {args.loss_3d_mid} | color: {(args.ratio_of_aug * 100):.0f}% | Max_count : {args.count} | Max_epoch : {args.epoch}")
        Graphormer_model, optimizer, batch_time, best_loss = train(args, trainset_loader, testset_loader, _model, epoch, best_loss, len(train_dataset), count, writer, pck_l, len(trainset_loader)+len(testset_loader), batch_time)
        loss, count, pck, batch_time = valid(args, trainset_loader, testset_loader, Graphormer_model, epoch, count, best_loss, len(train_dataset),  writer, batch_time, len(trainset_loader)+len(testset_loader), pck_l)
        
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
  

if __name__ == "__main__":
    args= parse_args()
    main(args)
