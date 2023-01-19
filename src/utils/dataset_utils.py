"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os.path as op
import numpy as np
import base64
import cv2
import yaml
from collections import OrderedDict
from scipy.linalg import orthogonal_procrustes


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


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
    t1 = pred.mean(0)
    pred_t = pred - t1

    # scale
    s1 = np.linalg.norm(pred_t) + 1e-8
    pred_t /= s1
    
    return pred_t + t1

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
    mtx2_t = np.dot(mtx2_t, R.T)
    # mtx2_t = np.dot(mtx2_t, R.T) * s
    # mtx2_t = mtx2_t * s1 + t1

    return mtx2_t + t1

