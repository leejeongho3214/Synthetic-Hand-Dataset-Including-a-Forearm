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
from tqdm import tqdm

def dump(pred_out_path, xyz_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list
            ], fo)
    # print('Dumped %d joints to %s' % (len(xyz_pred_list), pred_out_path))
    
def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss, T, dataset_name):
    pck_losses = AverageMeter()
    epe_losses = AverageMeter()
    Graphormer_model.eval()
    xyz_list = list()
    pred_path = os.path.join("eval_json", os.path.join(os.path.join(args.name[13:-31], dataset_name)))
    pred_out_path = os.path.join("eval_json", os.path.join(os.path.join(args.name[13:-31], dataset_name),  "pred.json"))
    if not os.path.isdir(pred_path): mkdir(pred_path)
    bbox_list = list()
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

                pck = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T, threshold = 'proportion')
                epe_loss, _ = EPE(pred_2d_joints, gt_2d_joint)
                pck_losses.update_p(pck * images.size(0), images.size(0))
                epe_losses.update_p(epe_loss[0], epe_loss[1])
                # for i in range(len(bbox)): bbox_list.append(int(bbox[i]))

                if T == 0.05 and iteration == 0:
                    for i in range(images.size(0)):
                        fig = plt.figure()
                        visualize_gt(images[i], gt_2d_joint[i], fig, i)
                        visualize_prediction(images[i], pred_2d_joints[i], fig, 'evaluation', epoch, i, args, dataset_name)
                        plt.close()
                xyz_list.append(pred_2d_joints)
        dump(pred_out_path, xyz_list)
        # plt.hist(bbox_list)
        # plt.xlabel('PCKb_length')
        # plt.ylabel('count')
        # plt.savefig("distriution.jpg")
        # print()
    
    else:
        heatmap_size, multiply = 64, 4
        if args.model == "hrnet": heatmap_size, multiply = 128, 2
        with torch.no_grad():
            for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):
                images = images.cuda()
                gt_2d_joint = gt_2d_joints.clone().detach()
                gt_2d_joint = gt_2d_joint.cuda()
                
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
                
                pck = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T, threshold = 'proportion')
                epe_loss,_ = EPE(pred_2d_joints, gt_2d_joint)
                pck_losses.update_p(pck * images.size(0), images.size(0))
                epe_losses.update_p(epe_loss[0], epe_loss[1])
                
                xyz_list.append(pred_2d_joints)
                if T == 0.05 and iteration == 0:
                    for i in range(images.size(0)):
                        fig = plt.figure()
                        visualize_gt(images[i], gt_2d_joint[i], fig, iteration)
                        visualize_prediction(images[i], pred_2d_joints[i], fig, 'evaluation', epoch, iteration, args, dataset_name)
                        plt.close()

        dump(pred_out_path, xyz_list)
        # plt.hist(bbox_list)
        # plt.xlabel('PCKb_length')
        # plt.ylabel('count')
        # plt.savefig("distriution.jpg")
        # print()
    return pck_losses.avg, epe_losses.avg
    

def main(args, T_list):
    root_path = "final_models"
    name_list = []
    loss = []
    other_list = ["14k_rot_color_1.0", "rot_color_0.6", "rot_color_frei"]
    
    for models_name in os.listdir(root_path):
        if models_name == "other_dataset": 
            for dataset_name in os.listdir(os.path.join(root_path, models_name)):
                name_list.append(os.path.join(os.path.join(root_path, models_name), dataset_name))
            continue
        
        if models_name == "ours":
            for ours_category in os.listdir(os.path.join(root_path, "ours")):
                if ours_category == "wrist":
                    current_loc = os.path.join(root_path,os.path.join(models_name, ours_category))
                    for kind in os.listdir(current_loc):
                        for aug in os.listdir(os.path.join(current_loc, kind)): name_list.append(os.path.join(os.path.join(current_loc, kind), aug))
                # else: 
                #     current_loc = os.path.join(os.path.join(os.path.join(root_path, "ours")), ours_category)
                #     for general_category in os.listdir(current_loc):
                #         name_list.append(os.path.join(current_loc, general_category))
                        
        else: 
            for a in other_list:
                name_list.append(os.path.join(os.path.join(root_path, models_name), a))
                continue
        
    name_list = ["final_models/ours/wrist/only_synthetic/rot_color_0.2", "final_models/ours/wrist/only_synthetic/rot_color_0.1"]    
    
    pbar = tqdm(total = len(name_list) * 4 * 4) 
    
    for name_p in name_list:
        sub_loss = []
        for T in T_list:         
            args.name = os.path.join(name_p, "checkpoint-good/state_dict.bin")
            args.model = args.name.split('/')[1]
            if args.model == "other_dataset": args.model = "ours"
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
                pck, epe = test(args, data_loader, _model, 0, 0, best_loss, T, set_name)    
                category_loss.append([set_name, pck * 100, epe * 0.26, args.name[13:-31]])    
                pbar.update(1)
            sub_loss.append(category_loss)
        loss.append(sub_loss)
    pbar.close()
    return loss

if __name__ == "__main__":
    args = parse_args()
    T_list = [0]
    losses = main(args, T_list)
    print("End")
