import argparse
import gc
import json
import os
import time
import os.path as op
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
import torch
import torchvision.models as models
from torch.utils import data
from argparser import load_model, parse_args
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  #
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
# sys.path.append("C:\\Users\\jeongho\\PycharmProjects\\PoseEstimation\\HandPose\\MeshGraphormer-main")
from dataset import *

from loss import *
from src.datasets.build import make_hand_data_loader
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.utils.comm import get_rank
from src.utils.geometric_layers import *
from src.utils.logger import setup_logger
from src.utils.metric_logger import AverageMeter
from src.utils.miscellaneous import mkdir
from visualize import *
        
def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss, T, dataset_name):
    pck_losses = AverageMeter()
    globals()['epe_p_21_losses'] = AverageMeter()
    for i in range(21): 
        globals()[f'epe_p_{i}_losses'] = AverageMeter()
    bbox_list = []
    with torch.no_grad():
        for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):

            Graphormer_model.eval()
            batch_size = images.size(0)
            images = images.cuda()
            gt_2d_joints = gt_2d_joints
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()

            pred_2d_joints = Graphormer_model(images)
            # pred_2d_joints = pred_3d_joints[:, :, :-1]

            pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)

            correct, visible_point, thresh = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, images, T, threshold='proportion') ## now, wrsit joint is excluded so if you want, change 1 to 0 into ran
            # correct, visible_point, thresh, bbox = PCK_2d_loss(pred_2d_joints, gt_2d_joint, images, T, threshold='proportion')
            epe_loss, epe_per = EPE(pred_2d_joints, gt_2d_joint)
            for i in range(1, 21): ## this excluded the wrist joint
                if epe_per[f'{i}'][1] == 0:
                    continue
                globals()['epe_p_{}_losses'.format(i)].update_p(epe_per[f'{i}'][0] * epe_per[f'{i}'][1], epe_per[f'{i}'][1])
            pck_losses.update_p(correct, visible_point)
            globals()['epe_p_21_losses'].update_p(epe_loss[0], epe_loss[1])
            # bbox_list.append(int(bbox[0]))

            if T == 0.05:
                fig = plt.figure()
                visualize_gt(images, gt_2d_joint, fig, iteration)
                visualize_prediction(images, pred_2d_joints, fig, 'evaluation', epoch, iteration, args, dataset_name)
                plt.close()

    # plt.hist(bbox_list,  color = "lightblue", ec="red" )
    # plt.savefig(f'{dataset_name}_distibution.jpg')
    # for i in range(21):
    #     print("{0}st epe error => {1:.2f}".format(i, globals()['epe_p_{}_losses'.format(i)].avg))

    # print( [round(globals()['epe_p_{}_losses'.format(i)].avg * 0.26, 2) for i in range(1, 22)])
    del test_dataloader
    gc.collect()


    return pck_losses.avg, globals()['epe_p_21_losses'].avg, thresh

def main(args, T):
    args.name = "output/general/rot_color/checkpoint-good/state_dict.bin"

    _model, logger, best_loss, epo, count = load_model(args)
    state_dict = torch.load(args.name)
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    _model.to(args.device)

    

    

    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                
    path = "../../datasets/0001_neutral_rigid"
    image_path = os.listdir(path)
    count = 0 
    for a in image_path:
        b = os.listdir(os.path.join(path, a))
        for index, name in enumerate(b):
            image = Image.open(os.path.join(path, os.path.join(a, name)))
            trans_image = trans(image)
            trans_image = trans_image[None,:,:,:].cuda()
            pred_2d_joints = _model(trans_image)
            pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * 224 ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * 224
            fig = plt.figure()
            visualize_prediction(trans_image, pred_2d_joints, fig, 'evaluation', 0, count, args, None)
            count += 1
            plt.close()
        


if __name__ == "__main__":
    args = parse_args()
    main(args, T=0.05)


