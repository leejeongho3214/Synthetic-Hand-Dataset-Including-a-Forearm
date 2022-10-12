import argparse
import json

import os
import math
import cv2
import mediapipe as mp
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import sys
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from src.tools.visualize import visualize_gt, visualize_prediction
from src.utils.drewing_utils import *
from loss import PCK_2d_loss_No_batch, MPJPE
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--multiscale_inference", default=False, action='store_true', )
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument("--aml_eval", default=False, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X steps.")
    parser.add_argument("--resume_path", default='HIU', type=str)
    #############################################################################################
            ## Set hyper parameter ##
    #############################################################################################
    parser.add_argument("--loss_2d", default=1, type=float,)
    parser.add_argument("--loss_3d", default=1, type=float,
                        help = "it is weight of 3d regression and '0' mean only 2d joint regression")
    parser.add_argument("--train", default='train', type=str, choices=['pre-train, train, fine-tuning'],
                        help = "3 type train method")
    parser.add_argument("--name", default='HIU_DMTL_full',
                        help = '20k means CISLAB 20,000 images',type=str)
    parser.add_argument("--root_path", default=f'output', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--output_path", default='HIU', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--iter", action='store_true')
    parser.add_argument("--iter2", action='store_true')
    parser.add_argument("--resume", action='store_true')
    #############################################################################################

    #############################################################################################
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.5, type=float)
    parser.add_argument("--vloss_w_sub", default=0.5, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='../../samples/unity/images/train/Capture0', type=str,
                        help="test data")
    parser.add_argument("--train_yaml", default='../../datasets/freihand/train.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--val_yaml", default='../../datasets/freihand/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--model_name_or_path", default='../modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                             "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2048,512,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--which_gcn", default='0,0,1', type=str,
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv")
    parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand")
    parser.add_argument("--run_eval_only", default=True, action='store_true', )
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    args = parser.parse_args()
    return args

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
# For static images:
path = '../../datasets/our_testset/1/rgb'
anno =  '../../datasets/our_testset/1/annotation'
IMAGE_FILES = os.listdir(path)

def main(T):
    args = parse_args()
    correct = 0
    visible_point = 0
    mp = 0
    bat = 0
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
      for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        # image = cv2.flip(cv2.imread(os.path.join(path, file)), 1)
        imagea = Image.open(os.path.join(path, file))
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_image = trans(imagea)[(2, 1, 0), :, :]

        image = cv2.imread(os.path.join(path, file))
        scale_x = 224 / image.shape[1]
        scale_y = 224 / image.shape[0]
        with open(os.path.join(anno, file)[:-4]+'.json', "r") as st_json:
            json_data = json.load(st_json)
            joint_total = json_data['annotations']
            joint = {}
            gt_2d = []
            for j in joint_total:
                if j['label'] != 'Pose':
                    if len(j['metadata']['system']['attributes']) > 0:
                        # Change 'z' to 'indicator function'
                        # Ex. 0 means visible joint, 1 means invisible joint
                        j['coordinates']['z'] = 0
                        joint[f"{int(j['label'])}"] = j['coordinates']
                    else:
                        j['coordinates']['z'] = 1
                        joint[f"{int(j['label'])}"] = j['coordinates']

            if len(joint) < 21:
                assert f"This {idx}.json is not correct"

            for h in range(0, 21):
                gt_2d.append([joint[f'{h}']['x'], joint[f'{h}']['y'], joint[f'{h}']['z']])
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

        # Print handedness (left v.s. right hand).
        # print(f'Handedness of {file}:')
        # print(results.multi_handedness)

        if not results.multi_hand_landmarks:
            continue
        # Draw hand landmarks of each hand.
        # print(f'Hand landmarks of {file}:')
        image_hight, image_width, _ = image.shape
        annotated_image = cv2.flip(image.copy(), 1)

        if len(results.multi_hand_landmarks) > 1:
            assert "This has two-hands"
        joint_2d = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Print index finger tip coordinates.
            # print(
            #     f'Index finger tip coordinate: (',
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            # )
            # I change this function from original resolution to 224 x 224
            joint = draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            for i in range(21):
                joint_2d.append(joint[i])
            gt_2d = torch.tensor(gt_2d)
            gt_2d[:,0] = gt_2d[:,0] * scale_x
            gt_2d[:, 1] = gt_2d[:, 1] * scale_y
            joint_2d = torch.tensor(joint_2d)
            joint_2d[:,0] = 224 - joint_2d[:,0]
            correct_, visible_point_ = PCK_2d_loss_No_batch(joint_2d, gt_2d ,image, T, file)

            mpjpe, batch = MPJPE(joint_2d.view(1,21,2), gt_2d.view(1,21,3))
            mp += mpjpe
            bat += batch
            args.name[7:-31] = 'mediapipe'
            if idx % 50 == 1:
                fig = plt.figure()
                visualize_gt(trans_image.unsqueeze(0), gt_2d.unsqueeze(0), fig)
                visualize_prediction(trans_image.unsqueeze(0), joint_2d.unsqueeze(0), fig, 'evaluation', idx, args)
                plt.close()

            correct += correct_
            visible_point += visible_point_

    # resize_and_show(cv2.flip(annotated_image, 1))
    print("Model_Name = MediePipe // Threshold = {} // pck===> {:.2f}% // mpjpe ====> {:.2f}".format(T, (correct/visible_point)*100, mp/bat))
    print(mp, bat)

if __name__ == '__main__':
    main(T=0.1)
    main(T=0.2)
    main(T=0.3)
    main(T=0.4)
    main(T=0.5)
