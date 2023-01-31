import sys
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.miscellaneous import mkdir
from src.utils.comm import is_main_process
from src.datasets.build import make_hand_data_loader
import json
import math
import torch
import os.path as op
import random
# from src.utils.dart_loader import DARTset
import cv2
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from torchvision import transforms
from src.utils.dataset_utils import align_scale, align_scale_rot

np.random.seed(77)


def build_dataset(args):   
    general_path = "../../datasets/general"
  
    args.dataset = args.name.split("/")[1]
    
    standard_j =  [[1.8155813217163086, 0.15561437606811523, 1.1083018779754639], [2.406423807144165, 0.5383367538452148, 1.304732084274292], [2.731782913208008, 1.172149658203125, 1.335669994354248], [2.681248903274536, 1.7862586975097656, 1.2639415264129639], [2.3304858207702637, 2.234518527984619, 1.1211540699005127], [2.341385841369629, 1.37321138381958, 2.0816190242767334], [2.3071250915527344, 2.0882482528686523, 1.7858655452728271], [2.2974867820739746, 2.293468952178955, 1.3347842693328857], [2.31135630607605, 1.9055771827697754, 1.029522180557251], [1.851935863494873, 1.30698823928833, 2.1360342502593994], [1.8758153915405273, 2.124051094055176, 2.5652201175689697], [1.973258376121521, 2.431856632232666, 2.1032679080963135], [2.0731117725372314, 2.644174098968506, 1.616095781326294], [1.471063256263733, 1.2448792457580566, 2.0854008197784424], [1.4334478378295898, 1.9523506164550781, 1.5718071460723877], [1.6441740989685059, 1.7141218185424805, 1.1860997676849365], [1.760351300239563, 1.242896556854248, 1.305544137954712], [1.1308115720748901, 1.1045317649841309, 1.9674842357635498], [1.0435627698898315, 1.6727776527404785, 1.8200523853302002], [1.2601540088653564, 1.6069226264953613, 1.4762027263641357], [1.4999980926513672, 1.5507283210754395, 1.1099226474761963]]
    standard_j = torch.tensor(standard_j)

    if args.dataset == "ours":
        train_path = os.path.join(general_path, "annotations/train")
        eval_path = os.path.join(general_path, "annotations/val")
        train_dataset = CustomDataset_g(args, train_path, standard_j)
        test_dataset = val_g_set(args, eval_path, standard_j)
        
    elif args.dataset == "frei":
        train_dataset = make_hand_data_loader(
            args, args.train_yaml, False, is_train=True, scale_factor=args.img_scale_factor, s_j = standard_j) 
        test_dataset = make_hand_data_loader(
            args, args.val_yaml, False, is_train=False, scale_factor=args.img_scale_factor, s_j = standard_j) 
        
        if args.dataset == "both":
            args.ratio_of_our = args.ratio_of_add
            o_dataset = CustomDataset_g(args, general_path + "/annotations/train", standard_j)
            train_dataset = ConcatDataset([train_dataset, o_dataset])
            
    elif args.dataset == "dart":
        train_dataset = DARTset(data_split='train')
        test_dataset = DARTset(data_split='test')
        
    else:
        assert 0, "you type the wrong dataset name"
        
    return train_dataset, test_dataset

    

class CustomDataset_g(Dataset):
    def __init__(self, args, path, standard_j):
        self.args = args
        self.noise_factor = 0.4
        self.path = path
        self.phase = path.split("/")[-1]
        self.root = "/".join(path.split("/")[:-2])
        with open(f"{path}/CISLAB_{self.phase}_data_update.json", "r") as st_json:
             self.meta = json.load(st_json)
        self.s_j = standard_j
        self.img_path = os.path.join(self.root, f"images/{self.phase}")
        self.img_res = 224
        self.ratio = 0.9
        self.scale_factor = 0.25
        self.rot_factor = 90 
        
    def __len__(self):
        return int(len(self.meta) * self.ratio)
        
    def __getitem__(self, idx):
        image, scale, rot, move_x, move_y = self.img_aug(idx)
        image = self.img_preprocessing(idx, image)
            
        image = torch.from_numpy(image).float()
        transformed_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
        joint_2d, joint_3d = self.joint_processing(idx, scale, rot, move_x, move_y)
            
            
        return transformed_img, joint_2d, joint_3d



    def joint_processing(self, idx, scale, rot, move_x, move_y): 
        
        joint_3d = torch.tensor(self.meta[f"{idx}"]['joint_3d'])
        if self.args.center:
            joint_2d = np.array(self.meta[f"{idx}"]['joint_2d'])
            joint_2d[:, 0] = joint_2d[:, 0] + move_x; joint_2d[:, 1] = joint_2d[:, 1] + move_y
        else:
            joint_2d = np.array(self.meta[f"{idx}"]['joint_2d'])
        joint_2d = self.j2d_processing(joint_2d, scale, rot) if self.args.crop else joint_2d
        joint_2d = torch.tensor(joint_2d)
        
        self.s_j[:, 0] = - self.s_j[:, 0]   ## This joint is always same for unified rotation
        self.s_j = self.s_j- self.s_j[0, :]
        
        joint_3d[:, 0] = - joint_3d[:, 0]   ## change the left hand to right hand
        joint_3d = joint_3d - joint_3d[0, :]
            
        if self.args.set == "scale":
            joint_3d = align_scale(joint_3d)
            
        elif self.args.set =="scale_rot":
            joint_3d = align_scale_rot(self.s_j, joint_3d)      
            
        return joint_2d, joint_3d
    
    
    def j2d_processing(self, kp, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, (112, 112), scale, 
                                    [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        # kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # kp = kp.astype('float32')
        return kp


    def img_preprocessing(self, idx, rgb_img):

        # in the rgb image we add pixel noise in a channel-wise manner
        if self.phase == 'train':
            if idx < int(self.args.ratio_of_aug * self.__len__()):
                pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
            else: 
                pn = np.ones(3)
        else:
            pn = np.ones(3)
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        
        return rgb_img

    
    def img_aug(self, idx):
        name = self.meta[f"{idx}"]['file_name']
        move_x = self.meta[f"{idx}"]['move_x']
        move_y = self.meta[f"{idx}"]['move_y']       
        image = cv2.imread(os.path.join(self.img_path, name))  # PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.args.center:
            if self.args.nn: nn = cv2.INTER_LINEAR 
            else: nn = None
            translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
            image = cv2.warpAffine(image, translation, (self.img_res, self.img_res),
                                 borderMode= nn)
            
        scale = min(1+self.scale_factor,
                        max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
        
        if idx < int(self.args.ratio_of_aug * self.__len__()):
            rot = min(2*self.rot_factor,
                        max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
        else:
            rot = 0
        """ Maybe, when it needs to crop img, i will modify the below code"""

        if self.args.crop:
            image = crop(image, (112, 112), scale, [self.img_res, self.img_res], rot=rot)
        
        return image, scale, rot, move_x, move_y
        
class val_g_set(CustomDataset_g):
    def __init__(self,  *args):
        super().__init__(*args)
        self.ratio_of_aug = 0
        self.ratio_of_dataset = 1
        with open(os.path.join(self.path, "CISLAB_val_data_update.json"), "r") as st_json:
            self.meta = json.load(st_json)
        self.img_path = os.path.join(self.root,f"images/{self.phase}" )
        self.ratio = 1

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def apply(img, aug, num=1, scale=1.5):
    Y = [aug(img) for _ in range(num)]
    return Y



def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
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

class Json_transform(Dataset):
    def __init__(self, degree, path):
        self.degree = degree
        self.path = path
        self.degree = degree
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)
        self.root = f'{self.path}/{self.degree}/images/train'
        self.store_path = f'{self.path}/{self.degree}/annotations/train/CISLAB_train_data_update.json'

    def get_json_g(self):
        meta_list = self.meta['images'].copy()
        index = []
        pbar = tqdm(total = len(meta_list))
        k = dict()
        count = 0
        for idx, j in enumerate(meta_list):
            pbar.update(1)
            if j['camera'] == '0':
                index.append(idx)
                continue
            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(os.path.join(self.root, name))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            
            for i in range(21):
                a = np.dot(np.array(rot, dtype='float32'),
                           np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
                a[:2] = a[:2] / a[2]
                b = a[:2] * focal_length + 112
                b = torch.tensor(b)

                for u in b:
                    if u > 223 or u < 0:
                        index.append(idx)
                        flag = True
                        break
                if flag:
                    break

                if i == 0:  # 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue


            flag = False
            for o in joint_2d:
                if o[0] > 220 or o[1] > 220:
                    flag = True
                    index.append(idx)
                    break
            if flag:
                continue
            
            center_j = np.array(joint_2d.mean(0))
            move_x = 112 - center_j[0]
            move_y = 112 - center_j[1]
            k[f"{count}"] = {'joint_2d': joint_2d.tolist(), 'joint_3d':joint.tolist(), 'move_x': move_x, 'move_y': move_y, "file_name": name}
            count += 1


        with open(self.store_path, 'w') as f:
            json.dump(k, f)

        print(
            f"Done ===> {self.store_path}")
        
    def get_json(self):
        meta_list = self.meta['images'].copy()
        index = []
        pbar = tqdm(total = len(meta_list))
        for idx, j in enumerate(meta_list):
            pbar.update(1)
            if j['camera'] == '0':
                index.append(idx)
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(os.path.join(self.root, name))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            left_pixel, right_pixel = [79-112, -112], [174-112, -112]
            left_rot = math.cos(
                rad) * left_pixel[1] - math.sin(rad) * left_pixel[0] + 112
            right_rot = math.cos(
                rad) * right_pixel[1] - math.sin(rad) * right_pixel[0] + 112

            if left_rot > 0:
                move_y = left_rot

            elif right_rot > 0:
                move_y = right_rot

            else:
                move_y = 0
            move_y2 = random.uniform(0, 40)

            for i in range(21):
                a = np.dot(np.array(rot, dtype='float32'),
                           np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
                a[:2] = a[:2] / a[2]
                b = a[:2] * focal_length + 112
                b = torch.tensor(b)

                for u in b:
                    if u > 223 or u < 0:
                        index.append(idx)
                        flag = True
                        break
                if flag:
                    break

                if i == 0:  # 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue

            d = joint_2d.clone()
            x = joint_2d[:, 0] - 112
            y = joint_2d[:, 1] - 112
            joint_2d[:, 0] = math.cos(rad) * x + math.sin(rad) * y + 112
            joint_2d[:, 1] = math.cos(
                rad) * y - math.sin(rad) * x + 112 + move_y + move_y2

            flag = False
            for o in joint_2d:
                if o[0] > 223 or o[1] > 223:
                    flag = True
                    index.append(idx)
                    break
            if flag:
                continue
                    
            rot_image = i_rotate(ori_image, degrees, 0, move_y + move_y2, interpolation=False)
            rot_image = Image.fromarray(rot_image)
            
            j['joint_2d'] = d.tolist()
            j['joint_3d'] = joint.tolist()
            j['rot_joint_2d'] = joint_2d.tolist()
            j['degree'] = degrees
            j['move'] = move_y + move_y2

        count = 0
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        with open(self.store_path, 'w') as f:
            json.dump(self.meta, f)

        print(
            f"Done ===> {self.store_path}")
        
        
def save_checkpoint(model, args, epoch, optimizer, best_loss, count, ment, num_trial=10, logger=None):

    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}'.format(
        ment))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer,
                'best_loss': best_loss,
                'count': count,
                'model_state_dict': model_to_save.state_dict()}, op.join(checkpoint_dir, 'state_dict.bin'))
            break
        except:
            pass

    return model_to_save, checkpoint_dir

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
    
class Json_e(Json_transform):
    def __init__(self, phase):
        root = "datasets/general_2M"
        if phase == 'eval':
            try:
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)       
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
                
            except:
                root = "../../datasets/ArmoHand"
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)   
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
            
        elif phase == 'test':
                root = "../../../../../../data1/ArmoHand"
                with open(os.path.join(root, "annotations/test/CISLAB_test_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)   
                with open(os.path.join(root, "annotations/test/CISLAB_test_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/test/CISLAB_test_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/test")
                self.store_path = os.path.join(root, "annotations/test/test_data_update.json")
            
        elif phase == 'train':
            root = "../../datasets/general"
            with open(os.path.join(root, "annotations/train/CISLAB_train_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(root, "annotations/train/CISLAB_train_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(root, "annotations/train/CISLAB_train_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
                
            self.root = os.path.join(root, "images/train")
            self.store_path = os.path.join(root, "annotations/train/CISLAB_train_data_update.json")
            
        elif phase == 'val':
            root = "../../datasets/general"
            with open(os.path.join(root, "annotations/val/CISLAB_val_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(root, "annotations/val/CISLAB_val_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(root, "annotations/val/CISLAB_val_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
                
            self.root = os.path.join(root, "images/val")
            self.store_path = os.path.join(root, "annotations/val/CISLAB_val_data_update.json")
    
def main():
    with open(os.path.join("../../datasets/general", "annotations/val/CISLAB_val_data_update.json"), "r") as st_json:
        meta = json.load(st_json)
    Json_e(phase = "train").get_json_g()
    print("ENDDDDDD")
    
if __name__ == '__main__':
    main()
    
        
