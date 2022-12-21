import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3" 
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from torch.utils.data import ConcatDataset
import torch
from torch.utils import data
from argparser import parse_args
import sys
from dataset import *
from loss import *
from src.utils.geometric_layers import *
from src.utils.metric_logger import AverageMeter
from visualize import *
from tqdm import tqdm


def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss, T, dataset_name):
    pck_losses = AverageMeter()
    epe_losses = AverageMeter()

    pred_path = os.path.join("eval_json", os.path.join(os.path.join(args.name[13:-31], dataset_name),  "pred.json"))
    with open(pred_path, 'r') as fi:
        pred_json = json.load(fi)
        
    for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):
        pred = torch.tensor(pred_json[0][iteration])
        correct, visible_point, thresh = PCK_2d_loss_visible(pred, gt_2d_joints, T, threshold = 'proportion')
        epe_loss, epe_per = EPE(pred, gt_2d_joints)
        pck_losses.update_p(correct, visible_point)
        epe_losses.update_p(epe_loss[0], epe_loss[1])
        
    return pck_losses.avg, epe_losses.avg, thresh

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
                        
        else: 
            for a in other_list:
                name_list.append(os.path.join(os.path.join(root_path, models_name), a))
                continue
        
    # name_list = ["final_models/ours/wrist/only_synthetic/13k_rot_color_1.0"]
    categories = ['general', 'p', 't', 't+p']
    pbar = tqdm(total = len(name_list) * len(categories) * len(T_list))

    for name_p in name_list:
        sub_loss = []
        for set_name in categories: 
              
            args.name = os.path.join(name_p, "checkpoint-good/state_dict.bin")
            args.model = args.name.split('/')[1]
            if args.model == "other_dataset": args.model = "ours"
            # _model, _, best_loss, _, count = load_model(args)
            # state_dict = torch.load(args.name)
            # _model.load_state_dict(state_dict['model_state_dict'], strict=False)
            # _model.cuda()

            path = "../../datasets/our_testset"
            folder_path = os.listdir(path)
            
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
            pck_losses = AverageMeter()
            
            for T in T_list:   
                data_loader = data.DataLoader(dataset=globals()[f'dataset_{set_name}'], batch_size=args.batch_size, num_workers=0, shuffle=False)
                pck, epe ,thresh= test(args, data_loader, None, 0, 0, 0, T, set_name)    
                if T in [0.05, 0.1, 0.15, 0.2]:
                    category_loss.append([set_name, pck * 100, epe * 0.26, args.name[13:-31]])    
                pck_losses.update_p(pck * 100, 1)
                pbar.update(1)
            sub_loss.append([category_loss, pck_losses.avg])
        loss.append(sub_loss)
    pbar.close()
    return loss

if __name__ == "__main__":
    args = parse_args()
    T_list = [i / 1000 for i in range(4, 202, 2)] 

    losses = main(args, T_list)
    for idx, loss in enumerate(losses):
        for i in range(4):
            print("{}; {}; {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} ".format(loss[i][0][0][0],loss[i][0][0][3] ,
                                                                                   loss[i][0][0][1], loss[i][0][1][1], loss[i][0][2][1], loss[i][0][3][1], loss[0][0][0][2], loss[i][1]))
