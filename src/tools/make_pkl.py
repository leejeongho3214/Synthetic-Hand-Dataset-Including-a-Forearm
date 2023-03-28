from torchvision import transforms
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
sys.path.append('/home/jeongho/tmp/Wearable_Pose_Model')
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import cv2
import random
import os.path as op
import torch
import math
import json
from src.datasets.build import make_hand_data_loader
from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir
import pickle

try:
    from src.utils.dart_loader import DARTset
except:
    print("Not import dart")

np.random.seed(77)
np.set_printoptions(precision=6, suppress=True)


class Json_3d(Dataset):

    def __init__(self, phase):
        with open(f"../../datasets/without_bg_revision/annotations/{phase}/CISLAB_{phase}_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"../../datasets/without_bg_revision/annotations/{phase}/CISLAB_{phase}_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"../../datasets/without_bg_revision/annotations/{phase}/CISLAB_{phase}_data.json", "r") as st_json:
            self.meta = json.load(st_json)

        self.root = f'../../datasets/without_bg_revision/images/{phase}'
        self.store_path = os.path.join(f'../../datasets/without_bg_revision/annotations/{phase}', "revision_data.pkl")
        self.dict = []

    def j2d_processing(self, kp, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2]+1, (512/2, 512/2), scale,
                                   [512, 512], rot=r)
        return kp

    def get_json(self, num, store):
        pbar = tqdm(total=len(self.meta['images']))
        count = 0

        for idx, j in enumerate(self.meta['images']):
            pbar.update(1)
            if j['camera'] == '0':
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint_3d = torch.tensor(
                self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = torch.tensor(
                self.camera['0']['focal'][f'{camera}'][0])
            translation = torch.tensor(self.camera['0']['campos'][f'{camera}'])
            rot = torch.tensor(self.camera['0']['camrot'][f'{camera}'])

            calibrationed_joint = torch.einsum(
                'ij, kj -> ki', rot, (joint_3d - translation))
            calibrationed_joint[:, :2] = calibrationed_joint[:,
                                                             :2]/(calibrationed_joint[:, 2][:, None].repeat(1, 2))
            calibrationed_joint = calibrationed_joint[:,
                                                      :2] * focal_length + 256

            image_path = os.path.join(
                self.root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if any(joint[idx] < 50 or joint[idx] > 470 for joint in calibrationed_joint for idx in range(2)):
                continue

            loof_count = 0
            while loof_count < 5:
                r = min(2*90, max(-2*90, np.random.randn()*90))
                scale = min(1+0.25, max(1-0.25, np.random.randn()*0.25+1))
                joint_2d = self.j2d_processing(
                    np.array(calibrationed_joint), scale, r)
                if not any(joint[idx] < 50 or joint[idx] > 470 for joint in joint_2d for idx in range(2)):
                    break
                loof_count += 1
                if loof_count == 4:
                    break
            if loof_count == 4:
                continue

            joint_2d = np.array(joint_2d)
            bbox = [min(joint_2d[:, 0]), min(joint_2d[:, 1]), max(
                joint_2d[:, 0]), max(joint_2d[:, 1])]
            bbox_size = np.sqrt(
                (bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2)

            if store:
                cropped_img = crop(image, (256, 256), scale, [512, 512], rot = r)
                
                bg_path = "../../datasets/without_bg_revision/background"
                bg_list = os.listdir(bg_path)
                bg_len = len(bg_list)
                bg_img = cv2.imread(os.path.join(bg_path, bg_list[idx%bg_len]))
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = cv2.resize(bg_img, (512, 512))
                
                index = np.where((cropped_img[:, :, 0] == 0) & (cropped_img[:, :, 1] == 0) & (cropped_img[:, :, 2] == 0))
                cropped_img[index] = bg_img[index]
                
                img_root_list = self.root.split('/')
                img_root_list[3] = "new_Armo"
                img_root = '/'.join(img_root_list)
                new_img_name = os.path.join(img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:]))
                new_img_fold_path = os.path.join(img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:-1]))
                
                if not os.path.isdir(new_img_fold_path):
                    mkdir(new_img_fold_path)
                
                cv2.imwrite(new_img_name, cropped_img[:, :, (2, 1, 0)])
        
            else:
                self.dict.append({'bbox': bbox, 'bbox_size': bbox_size, 'file_name': self.meta['images'][idx]['file_name'], 'joint_2d': joint_2d.tolist(), 'joint_3d': joint_3d.tolist(), 'rot': int(r), 'scale': round(scale, 4)})
                
            if count == num:
                break
            
            count += 1
            pbar.update(1)
        
        if not store:
            with open(self.store_path, 'wb') as f:
                pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = res[0] * scale
    # h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        # t = np.linalg.inv(t)
        t_torch = torch.from_numpy(t)
        t_torch = torch.inverse(t_torch)
        t = t_torch.numpy()
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]
    if not rot == 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot)
        new_img = myimrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = scipy.misc.imresize(new_img, res)
    new_img = myimresize(new_img, [res[0], res[1]])
    return new_img


def myimrotate(img, angle, center=None, scale=1.0, border_value=0, auto_bound=False):
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated

def myimresize(img, size, return_scale=False, interpolation='bilinear'):

    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, (size[0],size[1]), interpolation=cv2.INTER_LINEAR)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    result = cv2.warpAffine(result, translation, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)

    return result

def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    result = cv2.warpAffine(result, translation, (new_w, new_h))

    return result

def visualize(image, gt_2d_joint):
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    
    for i in range(21):

        cv2.circle(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])), 2, [0, 1, 0],
                    thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])),
                        (int(gt_2d_joint[parents[i]][0]), int(gt_2d_joint[parents[i]][1])),
                        [0, 0, 1], 1)

    plt.imshow(image)
    plt.show()
    
def visualize_bbox(image, bbox):
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    cv2.line(image, (int(bbox[0]), int(bbox[1])),
                (int(bbox[0]), int(bbox[3])),
                [0, 0, 1], 4)
    cv2.line(image, (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[1])),
            [0, 0, 1], 4)
    cv2.line(image, (int(bbox[2]), int(bbox[3])),
        (int(bbox[0]), int(bbox[3])),
        [0, 0, 1], 4)
    cv2.line(image, (int(bbox[2]), int(bbox[3])),
    (int(bbox[2]), int(bbox[1])),
    [0, 0, 1], 4)

    plt.imshow(image)
    plt.show()


def main():
    img_store = False
    Json_3d(phase="train").get_json(-1, img_store)
    Json_3d(phase="val").get_json(-1, img_store)
    print("ENDDDDDD")


if __name__ == '__main__':
    main()
