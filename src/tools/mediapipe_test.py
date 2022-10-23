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
from argparser import parse_args, load_model
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  #
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
import mediapipe as mp
from src.utils.drewing_utils import *
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  plt.imshow(img[:,:,[2,1,0]])
  plt.show()


def media_test(args, test_dataloader, T, dataset_name):
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1) as hands:
        pck_losses = AverageMeter()
        mpjpe_losses = AverageMeter()
        count = 0
        for iteration, (images, gt_2d, annotated_image) in enumerate(test_dataloader):
            images = images[0]
            images = np.array(images)
            results = hands.process(cv2.flip(cv2.cvtColor(images, cv2.COLOR_BGR2RGB), 1))
            if not results.multi_hand_landmarks:
                continue
            # Draw hand landmarks of each hand.
            # print(f'Hand landmarks of {file}:')
            # image_hight, image_width = image.shape

            if len(results.multi_hand_landmarks) > 1:
                assert "This has two-hands"
            joint_2d = []
            iimage = images.copy()
            for hand_landmarks in results.multi_hand_landmarks:

                joint = draw_landmarks(
                    iimage,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for i in range(21):
                joint_2d.append(joint[i])

            joint_2d = torch.tensor(joint_2d)
            joint_2d[:,0] = 224 - joint_2d[:,0]
            pck, thresh = PCK_2d_loss_No_batch(joint_2d, gt_2d[0] ,images, T, threshold = 'pixel')
            pck_losses.update(pck, args.batch_size)

            mpjpe = MPJPE_visible(joint_2d.view(1,21,2), gt_2d.view(1,21,3))
            mpjpe_losses.update(mpjpe,args.batch_size)
            count += 1
            if T == 0.05:
                fig = plt.figure()
                visualize_gt_media(images, gt_2d, fig, iteration)
                visualize_prediction_media(images, joint_2d.unsqueeze(0), fig, 'evaluation', iteration,args, dataset_name)
                plt.close()
            if iteration == len(test_dataloader)-1:
                print(count)
        return pck_losses.avg, mpjpe_losses.avg, thresh


def main(args, T):
    count = 0
    args.name = "final_models/MediaPipe/checkpoint-good/state_dict.bin"

    path = "../../datasets/our_testset"
    folder_path = os.listdir(path)

    categories = ['general', 'out_of_bound', 'p', 't', 't+p']

    for name in categories:
        count = 0
        for  num in folder_path:
            if num[0] == '.':
                continue
            if count == 0:
                dataset = Our_testset_media(path, os.path.join(num,name))
                count += 1
                continue
            if count > 0:
                previous_dataset = Our_testset_media(path, os.path.join(num,name))
                globals()[f'dataset_{name}'] = ConcatDataset([previous_dataset, dataset])

    loss = []
    for set_name in categories:
        
        data_loader = data.DataLoader(dataset=globals()[f'dataset_{set_name}'], batch_size=args.batch_size, num_workers=0, shuffle=False)

        # pck, mpjpe = test(args, data_loader, _model, 0, 0, best_loss, T, set_name)
        pck, mpjpe, thresh = media_test(args, data_loader, T, set_name)
        if thresh == 'pixel':
            print("Model_Name = {}  // {} //Threshold = {} mm // pck===> {:.2f}% // mpjpe===> {:.2f}mm // {} // {}".format(args.name[13:-31], thresh, T * 20, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
        else:
            print("Model_Name = {}  // {} //Threshold = {} // pck===> {:.2f}% // mpjpe===> {:.2f}mm // {} // {}".format(args.name[13:-31], thresh, T, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
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
        print("dataset = {} ,{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, mpjpe = {:.2f}mm".format(i[0],loss[idx][1], loss1[idx][1], loss2[idx][1], loss3[idx][1], loss4[idx][1], loss5[idx][1], loss6[idx][1], loss7[idx][1], loss8[idx][1], loss9[idx][1], loss[idx][2]))