
import cv2
import argparse
import gc
import json
import os
import time
import os.path as op
import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
import torch
import torchvision.models as models
from torch.utils import data
from argparser import load_model, parse_args
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


args = parse_args()
args.name = "final_models/synthetic_aug_with_frei/rot_trans_color/checkpoint-good/state_dict.bin"
_model, logger, best_loss, epo = load_model(args)
state_dict = torch.load(args.name)
_model.load_state_dict(state_dict['model_state_dict'], strict=False)
_model.to(args.device)

cap = cv2.VideoCapture('../../datasets/demo3.mp4')
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# 1프레임과 다음 프레임 사이의 간격 설정
delay = round(1000/fps)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter('ours.avi', fourcc, fps, (224,224))
while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = trans(image)
    _model.eval()
    _model = _model.cuda()
    image = image.unsqueeze(0)
    image = image.cuda()
    pred_2d_joints = _model(image)
    pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * image.size(2) ## You Have to check whether weight and height is correct dimenstion
    pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * image.size(3)
    image = visualize_prediction_video(image, pred_2d_joints) ## Draw the 2d joint in image
    image = (image * 244).astype(np.uint8)
    # cv2.imshow('Model Hands',image)
    # image = np.unit16(image*255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()



# with mp_hands.Hands(
# static_image_mode=True,
# max_num_hands=1,
# min_detection_confidence=0.1) as hands:
#     cap = cv2.VideoCapture('../../datasets/demo3.mp4')
#     w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

#     # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#     # 1프레임과 다음 프레임 사이의 간격 설정
#     delay = round(1000/fps)

#     # 웹캠으로 찰영한 영상을 저장하기
#     # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
#     out = cv2.VideoWriter('mediapipe.avi', fourcc, fps, (224,224))
#     while cap.isOpened():
#         success, image = cap.read()
        
#         if not success:
#             print("Ignoring empty camera frame.")
#             # If loading a video, use 'break' instead of 'continue'.
#             break
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)

#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())
#             # Flip the image horizontally for a selfie-view display.
#         image = cv2.resize(image, (224,224))
#         out.write(image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#     cap.release()

