import json

import os
import math
import cv2
import mediapipe as mp
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from src.tools.visualize import visualize_gt, visualize_prediction
from src.utils.drewing_utils import *
from loss import PCK_2d_loss_No_batch

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
path = '../../datasets/our_testset/rgb'
anno =  '../../datasets/our_testset/annotation'
IMAGE_FILES = os.listdir(path)

def main(T):
    correct = 0
    visible_point = 0
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

            # fig = plt.figure()
            # visualize_gt(trans_image.unsqueeze(0), gt_2d.unsqueeze(0), fig)
            # visualize_prediction(trans_image.unsqueeze(0), joint_2d.unsqueeze(0), fig)
            # plt.close()

            correct += correct_
            visible_point += visible_point_

    # resize_and_show(cv2.flip(annotated_image, 1))
    print("Model_Name = MediePipe // Threshold = {} // pck===> {:.2f}%".format(T, (correct/visible_point)*100))

if __name__ == '__main__':
    main(T=0.1)
    main(T=0.2)
    main(T=0.3)
    main(T=0.4)
    main(T=0.5)
