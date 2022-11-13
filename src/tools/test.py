
import os
import random
from re import L
import torchvision
import cv2
import random
import math
from PIL import Image
from cv2 import illuminationChange
from d2l import torch as d2l
import numpy as np
aa = '/home/jeongho/tmp/Wearable_Pose_Model/datasets/org/0/images/train/Capture0/0'
bb = os.listdir(aa)

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    return Y

def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    translation = np.float32([[1,0,move_x], [0,1,move_y]])
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    result = cv2.warpAffine(result, translation, (new_w, new_h))
    
    return result

for idx, i in enumerate(bb):
    image = cv2.imread(os.path.join(aa, i))
    root = "../../datasets/background/bg"
    path = os.listdir(root)
    bg = cv2.imread(os.path.join(root, random.choice(path)))
    bg = cv2.resize(bg, (224,224))
    



    image[image < 30] = 0 ## Remove the noise, not hand pixel


    # # for idx1, i in enumerate(loc):
    # #     for idx2, j in enumerate(i):
    # #         if 20 < idx1 < 200 and 20 < idx2 <200 and j == True:
    # #             if image[idx1, idx2+1][0] == 0 or image[idx1, idx2-1][0] ==0 or image[idx1+1, idx2][0] ==0  or image[idx1-1, idx2][0] ==0 or image[idx1-1, idx2-1][0] ==0 or image[idx1-1, idx2+1][0] ==0 or image[idx1+1, idx2-1][0] ==0 or image[idx1+1, idx2+1][0] ==0:
    # #                 loc[idx1, idx2] = False
    # #             if image[idx1, idx2+1][1] == 0 or image[idx1, idx2-1][1] ==0 or image[idx1+1, idx2][1] ==0  or image[idx1-1, idx2][1] ==0 or image[idx1-1, idx2-1][1] ==0 or image[idx1-1, idx2+1][1] ==0 or image[idx1+1, idx2-1][1] ==0 or image[idx1+1, idx2+1][1] ==0:
    # #                 loc[idx1, idx2] = False
    # #             if image[idx1, idx2+1][2] == 0 or image[idx1, idx2-1][2] ==0 or image[idx1+1, idx2][2] ==0  or image[idx1-1, idx2][2] ==0 or image[idx1-1, idx2-1][2] ==0 or image[idx1-1, idx2+1][2] ==0 or image[idx1+1, idx2-1][2] ==0 or image[idx1+1, idx2+1][2] ==0:
    #                 # loc[idx1, idx2] = False




    # src_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(src_hsv, (0, 0, 0), (180, 100,0.1))


    degree = random.uniform(-20, 20)
    rad = math.radians(degree)
    left_pixel, right_pixel = [79-112, -112], [174-112, -112]
    left_rot = math.cos(rad) * left_pixel[1] - math.sin(rad) * left_pixel[0] + 112
    right_rot = math.cos(rad) * right_pixel[1] - math.sin(rad) * right_pixel[0] + 112

    if left_rot > 0:
        move_y = left_rot

    elif right_rot > 0:
        move_y = right_rot

    else:
        move_y = 0
    color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    move_x = random.uniform(-30, 30)
    y = random.uniform(0, 40)
    images  = i_rotate(image, degree, 0 , 0)
    imagess = i_rotate(image, degree, 0 , move_y + y)
    
    loc = np.all(imagess != [0, 0, 0], axis=-1)
    bg[loc] = [0, 0 ,0]
    imagessss = imagess + bg
    imag = apply(Image.fromarray(np.array(image)),color_aug,1,1)[0]
    iim = apply(Image.fromarray(np.array(imagess)),color_aug,1,1)[0]
    cv2.imwrite(f'ori/ori_{idx}.jpg', image)
    cv2.imwrite(f'rot/rot_{idx}.jpg', imagess)
    cv2.imwrite(f'color/color_{idx}.jpg', np.array(imag))
    cv2.imwrite(f'all/all_{idx}.jpg', np.array(iim))
    # cv2.imwrite(f'bg/bg_{idx}.jpg', imagessss)
    if idx == 20:
        break




    # cv2.imshow('ori_image',image)
    # cv2.moveWindow('ori_image',426,350)
    

    # cv2.imshow('rot',images)
    # cv2.moveWindow('rot',650,350)

    # cv2.imshow('rot_trans',imagess)
    # cv2.moveWindow('rot_trans',874,350)

    # cv2.imshow('rot_trans_bg',imagessss)
    # cv2.moveWindow('rot_trans_bg',1098,350)

    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 