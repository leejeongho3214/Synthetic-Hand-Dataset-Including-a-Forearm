import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torch
from torch.utils import data
from argparser import load_model, parse_args
import sys
from dataset import *
from loss import *
from src.utils.geometric_layers import *
from src.utils.metric_logger import AverageMeter
from visualize import *
    
def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss, T, dataset_name):
    pck_losses = AverageMeter()
    mpjpe_losses = AverageMeter()
    bbox_list = []

    if args.model == "ours":
        with torch.no_grad():
            for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):
                Graphormer_model.eval()
                
                images = images.cuda()
                gt_2d_joints = gt_2d_joints
                gt_2d_joint = gt_2d_joints.clone().detach()
                gt_2d_joint = gt_2d_joint.cuda()

                pred_2d_joints = Graphormer_model(images)

                pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)

                correct, visible_point, thresh = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, images, T, threshold = 'proportion')
                # correct, visible_point, thresh, bbox = PCK_2d_loss(pred_2d_joints, gt_2d_joint, images, T, threshold='proportion')
                mpjpe_loss = MPJPE_visible(pred_2d_joints, gt_2d_joint)
                pck_losses.update_p(correct, visible_point)
                mpjpe_losses.update(mpjpe_loss, args.batch_size)
                # bbox_list.append(int(bbox[0]))

                if T == 0.05:
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig, iteration)
                    visualize_prediction(images, pred_2d_joints, fig, 'evaluation', epoch, iteration, args, dataset_name)
                    plt.close()

        return pck_losses.avg, mpjpe_losses.avg, thresh
    
    else:
        heatmap_size, multiply = 64, 4
        if args.model == "hrnet": heatmap_size, multiply = 128, 2
        for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):
            images = images.cuda()
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()
             
            if args.model == "hourglass": images = images.permute(0,1,3,2)
            pred = Graphormer_model(images)
            if args.model == "hourglass": pred = pred[:, -1]

            pred_joint = np.zeros((pred.size(0),pred.size(1),2))
            for idx, batch in enumerate(pred):
                for idx2, joint in enumerate(batch):
                    joint = joint.detach().cpu()
                    joint = joint.flatten()
                    index = joint.argmax()
                    row = int(index / heatmap_size)
                    col = index % heatmap_size
                    pred_joint[idx,idx2] = np.array([col, row]).flatten()
            pred_joint = torch.tensor(pred_joint)
            pred_2d_joints = pred_joint * multiply ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
            
            
            correct, visible_point, thresh = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, images, T, threshold='proportion')
            # correct, visible_point, thresh, bbox = PCK_2d_loss(pred_2d_joints, gt_2d_joint, images, T, threshold='proportion')
            mpjpe_loss = MPJPE_visible(pred_2d_joints, gt_2d_joint)
            pck_losses.update_p(correct, visible_point)
            mpjpe_losses.update(mpjpe_loss, args.batch_size)
            # bbox_list.append(int(bbox[0]))

            if T == 0.05:
                if args.model == "hourglass": images = images.permute(0,1,3,2)
                fig = plt.figure()
                visualize_gt(images, gt_2d_joint, fig, iteration)
                visualize_prediction(images, pred_2d_joints, fig, 'evaluation', epoch, iteration, args, dataset_name)
                plt.close()

    return pck_losses.avg, mpjpe_losses.avg, thresh
    


def main(args, T):
    args.name = "final_models/hrnet/rot_color_frei/checkpoint-good/state_dict.bin"
    args.model = args.name.split('/')[1]
    if args.model == "hourglass": args.batch_size = 16
    _model, _, best_loss, _, count = load_model(args)
    state_dict = torch.load(args.name)
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    _model.cuda()

    path = "../../datasets/our_testset"
    folder_path = os.listdir(path)

    categories = ['general', 'p', 't', 't+p']
    for name in categories:
        count = 0
        for num in folder_path:
            if num[0] == '.' or not os.path.isdir(f"{path}/{num}/{name}"):
                continue
            if count == 0:
                dataset = Our_testset(path, os.path.join(num,name), args.model)
                count += 1
                continue
            if count > 0:
                previous_dataset = Our_testset(path, os.path.join(num,name), args.model)
                dataset = ConcatDataset([previous_dataset, dataset])
        globals()[f'dataset_{name}'] = dataset
        
    loss = []
    for set_name in categories: 
        data_loader = data.DataLoader(dataset=globals()[f'dataset_{set_name}'], batch_size=args.batch_size, num_workers=0, shuffle=False)
        pck, mpjpe ,thresh= test(args, data_loader, _model, 0, 0, best_loss, T, set_name)
        
        if thresh == 'pixel':
            print("Model_Name = {}  // {} //Threshold = {} mm // pck===> {:.2f}% // mpjpe===> {:.2f}mm // {} // {}".format(args.name[13:-31],thresh, T * 20, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
        else:
            print("Model_Name = {}  // Threshold = {} // pck===> {:.2f}% // mpjpe===> {:.2f}mm // {} // {}".format(args.name[13:-31], T, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
        loss.append([set_name, pck * 100, mpjpe * 0.26])

    print("==" * 80)
    return loss


if __name__ == "__main__":
    args = parse_args()
    loss = main(args, T=0.05)
    loss1 = main(args, T=0.1)
    loss2 = main(args, T=0.15)
    loss3 = main(args, T=0.2)
    loss4 = main(args, T=0.25)
    loss5 = main(args, T=0.3)
    loss6 = main(args, T=0.35)
    loss7 = main(args, T=0.4)
    loss8 = main(args, T=0.45)
    loss9 = main(args, T=0.5)
    for idx,i in enumerate(loss):
        print("dataset = {} ,{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(i[0],loss[idx][1], loss1[idx][1], loss2[idx][1], loss3[idx][1], loss4[idx][1], loss5[idx][1], loss6[idx][1], loss7[idx][1], loss8[idx][1], loss9[idx][1], loss[idx][2]))
