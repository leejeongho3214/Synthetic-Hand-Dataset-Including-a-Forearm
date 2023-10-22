"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import sys
import torch
import os.path as op
import numpy as np
import base64
import cv2
import yaml
from collections import OrderedDict
from scipy.linalg import orthogonal_procrustes
np.random.seed(77)

def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None
class GenerateHeatmap:
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(self, p):
        hms = np.zeros(
            shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32
        )
        sigma = self.sigma
        for idx, pt in enumerate(p):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d]
                )
        return hms
    
def load_labelmap(labelmap_file):
    label_dict = None
    if labelmap_file is not None and op.isfile(labelmap_file):
        label_dict = OrderedDict()
        with open(labelmap_file, 'r') as fp:
            for line in fp:
                label = line.strip().split('\t')[0]
                if label in label_dict:
                    raise ValueError("Duplicate label " + label + " in labelmap.")
                else:
                    label_dict[label] = len(label_dict)
    return label_dict


def load_shuffle_file(shuf_file):
    shuf_list = None
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            shuf_list = []
            for i in fp:
                shuf_list.append(int(i.strip()))
    return shuf_list


def load_box_shuffle_file(shuf_file):
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            img_shuf_list = []
            box_shuf_list = []
            for i in fp:
                idx = [int(_) for _ in i.strip().split('\t')]
                img_shuf_list.append(idx[0])
                box_shuf_list.append(idx[1])
        return [img_shuf_list, box_shuf_list]
    return None

def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)
    
def align_scale(pred):   ## mtx2 is pred
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    # t1 = pred.mean(0)
    # pred_t = pred - t1

    # s1 = np.sqrt(np.square(pred[9] - pred[8]).sum())
    # pred /= s1 + sys.epsilon

    # 1. original method
    # s1 = np.linalg.norm(pred) + 1e-8
    # pred /= s1
    
    # 2. new method to make this vector norm about from 0.8 to 1.2
    # s1 = np.linalg.norm(pred) + 1e-8
    # pred /= s1
    # pred /= (1 + np.float(np.random.normal(0, 0.1 , 1)))
    
    # 3. method to normalized this vector through middle joint length (8-9 number joint)
    s1 = np.sqrt(np.square(pred[9] - pred[8]).sum())
    pred /= s1 + sys.float_info.epsilon

    return pred

def align_scale_rot(mtx1, mtx2):   ## mtx2 is pred
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = torch.tensor(np.dot(mtx2_t, R.T)) + t1
    # mtx2_t = np.dot(mtx2_t, R.T) * s
    # mtx2_t = mtx2_t * s1 + t1

    return mtx2_t 

 