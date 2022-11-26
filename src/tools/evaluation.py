import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
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
    epe_losses = AverageMeter()
    Graphormer_model.eval()
    
    if args.model == "ours":
        with torch.no_grad():
            for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):
                images = images.cuda()
                gt_2d_joints = gt_2d_joints
                gt_2d_joint = gt_2d_joints.clone().detach()
                gt_2d_joint = gt_2d_joint.cuda()

                pred_2d_joints = Graphormer_model(images)

                pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)

                correct, visible_point, thresh = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T, threshold = 'proportion')
                epe_loss, epe_per = EPE(pred_2d_joints, gt_2d_joint)
                pck_losses.update_p(correct, visible_point)
                epe_losses.update_p(epe_loss[0], epe_loss[1])
                # bbox_list.append(int(bbox[0]))

                if T == 0.05:
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig, iteration)
                    visualize_prediction(images, pred_2d_joints, fig, 'evaluation', epoch, iteration, args, dataset_name)
                    plt.close()

        return pck_losses.avg, epe_losses.avg, thresh
    
    else:
        heatmap_size, multiply = 64, 4
        if args.model == "hrnet": heatmap_size, multiply = 128, 2
        with torch.no_grad():
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
                
                correct, visible_point, thresh = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T, threshold = 'proportion')
                epe_loss, epe_per = EPE(pred_2d_joints, gt_2d_joint)
                pck_losses.update_p(correct, visible_point)
                epe_losses.update_p(epe_loss[0], epe_loss[1])

                if T == 0.05:
                    if args.model == "hourglass": images = images.permute(0,1,3,2)
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig, iteration)
                    visualize_prediction(images, pred_2d_joints, fig, 'evaluation', epoch, iteration, args, dataset_name)
                    plt.close()

    return pck_losses.avg, epe_losses.avg, thresh
    

def main(args, T_list):
    root_path = "final_models"
    name_list = []
    loss = []
    
    for models_name in os.listdir(root_path):
        if models_name == "other_dataset": 
            for dataset_name in os.listdir(os.path.join(root_path, models_name)):
                name_list.append(os.path.join(os.path.join(root_path, models_name), dataset_name))
            continue
        if models_name == "ours":
            current_loc = os.path.join(root_path, "ours/wrist")
            for kind in os.listdir(current_loc):
                for aug in os.listdir(os.path.join(current_loc, kind)): name_list.append(os.path.join(os.path.join(current_loc, kind), aug))
        else: name_list.append(os.path.join(os.path.join(root_path, models_name), "rot_color_frei")); continue
        
    name_list = ["final_models/other_dataset/frei_6k"]    
        
    for name_p in name_list:
        sub_loss = []
        for T in T_list:         
            args.name = os.path.join(name_p, "checkpoint-good/state_dict.bin")
            args.model = args.name.split('/')[1]
            if args.model == "hourglass": args.batch_size = 16
            elif args.model == "other_dataset": args.model = "ours"
            _model, _, best_loss, _, count = load_model(args)
            state_dict = torch.load(args.name)
            _model.load_state_dict(state_dict['model_state_dict'], strict=False)
            _model.cuda()


            path = "../../datasets/our_testset"
            folder_path = os.listdir(path)
            
            categories = ['general', 'p', 't', 't+p']
            category_loss = []
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
                

            for set_name in categories: 
                data_loader = data.DataLoader(dataset=globals()[f'dataset_{set_name}'], batch_size=args.batch_size, num_workers=0, shuffle=False)
                pck, epe ,thresh= test(args, data_loader, _model, 0, 0, best_loss, T, set_name)
                
                if thresh == 'pixel':
                    print("Model_Name = {}  // {} //Threshold = {} mm // pck===> {:.2f}% // epe===> {:.2f}mm // {} // {}".format(args.name[13:-31],thresh, T * 20, pck * 100, epe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
                else:
                    print("Model_Name = {}  // Threshold = {} // pck===> {:.2f}% // epe===> {:.2f}mm // {} // {}".format(args.name[13:-31], T, pck * 100, epe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
                    
                category_loss.append([set_name, pck * 100, epe * 0.26, args.name[13:-31]])    
            sub_loss.append(category_loss)
            print("==" * 60)
        loss.append(sub_loss)
    return loss

if __name__ == "__main__":
    args = parse_args()
    losses = main(args, T_list=[0.05, 0.1, 0.15, 0.2])
    for idx,loss in enumerate(losses):
        for i in range(4):
            print("Category = {} ,{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} // Model_Name = {}".format(loss[0][i][0],loss[0][i][1], loss[1][i][1], loss[2][i][1], loss[3][i][1], loss[0][i][2], loss[0][i][3]))
