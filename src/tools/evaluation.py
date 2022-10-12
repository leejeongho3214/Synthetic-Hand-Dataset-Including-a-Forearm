import argparse
import gc
import json
import os
import time
import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
import torch
import torchvision.models as models
from torch.utils import data
from argparser import parse_args, load_model
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  #
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


        
def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss, T):

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    pck_losses = AverageMeter()
    mpjpe_losses = AverageMeter()

    with torch.no_grad():
        for iteration, (images, gt_2d_joints) in enumerate(test_dataloader):

            Graphormer_model.eval()
            batch_size = images.size(0)
            images = images.cuda()
            gt_2d_joints = gt_2d_joints
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()

            pred_camera, pred_3d_joints = Graphormer_model(images)
            if args.loss_3d == 0:
                pred_2d_joints = pred_3d_joints[:, :, :-1]

            else:
                pred_2d_joints = orthographic_projection(pred_3d_joints.contiguous(), pred_camera.contiguous())

            pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)

            correct, visible_point = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, images, T, threshold='pixel')
            mpjpe_loss = MPJPE_visible(pred_2d_joints, gt_2d_joint)
            pck_losses.update_p(correct, visible_point)
            mpjpe_losses.update(mpjpe_loss, args.batch_size)

            fig = plt.figure()
            visualize_gt(images, gt_2d_joint, fig)
            visualize_prediction(images, pred_2d_joints, fig, 'evaluation', iteration,args)
            plt.close()

    del test_dataloader
    gc.collect()

    return pck_losses.avg, mpjpe_losses.avg

def main(args, T):
    count = 0
    args.name = "output/new_synthetic/only_2d/checkpoint-good/state_dict.bin"
    _model, logger, best_loss, epo = load_model(args)
    state_dict = torch.load(args.name)
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    _model.to(args.device)

    folder_path = os.listdir("../../datasets/our_testset")
    count = 0
    for idx, num in enumerate(folder_path):
        if num[0] == '.':
            continue
        if count == 0:
            dataset = Our_testset(num)
        count += 1
        if count > 1:
            previous_dataset = Our_testset(num)
            dataset = ConcatDataset([previous_dataset, dataset])

    data_loader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    pck, mpjpe = test(args, data_loader, _model, 0, 0, best_loss, T)

    print("Model_Name = {}  // Threshold = {} // pck===> {:.2f}% // mpjpe===> {:.2f}mm // {}".format(args.name[7:-31], T, pck * 100, mpjpe * 0.26, len(dataset)))
    gc.collect()
    torch.cuda.empty_cache()

    return pck * 100, mpjpe * 0.26


if __name__ == "__main__":
    args = parse_args()
    loss, mp = main(args, T=0.1)
    loss1, _ =main(args, T=0.2)
    loss2, _ = main(args, T=0.3)
    loss3, _ = main(args, T=0.4)
    loss4, _ =main(args, T=0.5)
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(loss, loss1, loss2, loss3, loss4))
    print(mp)