# import pickle
import pickle5 as pickle
import sys
from matplotlib import pyplot as plt
import os
from src.utils.dataset_loader import GAN, SyntheticHands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.miscellaneous import mkdir
from src.utils.comm import is_main_process
from src.datasets.build import make_hand_data_loader
import torch
import os.path as op
try:
    from src.utils.dart_loader import DARTset
except:
    print("Not import dart")
import cv2
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, random_split
from PIL import Image
from torchvision import transforms

np.random.seed(77)
np.set_printoptions(precision=6, suppress=True)

def build_dataset(args):   
    general_path = "../../datasets/without_bg_revision"
    args.dataset = args.name.split("/")[1]
    
    standard_j =  [[1.8155813217163086, 0.15561437606811523, 1.1083018779754639], [2.406423807144165, 0.5383367538452148, 1.304732084274292], [2.731782913208008, 1.172149658203125, 1.335669994354248], [2.681248903274536, 1.7862586975097656, 1.2639415264129639], [2.3304858207702637, 2.234518527984619, 1.1211540699005127], [2.341385841369629, 1.37321138381958, 2.0816190242767334], [2.3071250915527344, 2.0882482528686523, 1.7858655452728271], [2.2974867820739746, 2.293468952178955, 1.3347842693328857], [2.31135630607605, 1.9055771827697754, 1.029522180557251], [1.851935863494873, 1.30698823928833, 2.1360342502593994], [1.8758153915405273, 2.124051094055176, 2.5652201175689697], [1.973258376121521, 2.431856632232666, 2.1032679080963135], [2.0731117725372314, 2.644174098968506, 1.616095781326294], [1.471063256263733, 1.2448792457580566, 2.0854008197784424], [1.4334478378295898, 1.9523506164550781, 1.5718071460723877], [1.6441740989685059, 1.7141218185424805, 1.1860997676849365], [1.760351300239563, 1.242896556854248, 1.305544137954712], [1.1308115720748901, 1.1045317649841309, 1.9674842357635498], [1.0435627698898315, 1.6727776527404785, 1.8200523853302002], [1.2601540088653564, 1.6069226264953613, 1.4762027263641357], [1.4999980926513672, 1.5507283210754395, 1.1099226474761963]]
    standard_j = torch.tensor(standard_j)
        
    if args.dataset == "frei":
        train_dataset = make_hand_data_loader(
            args, args.train_yaml, False, is_train=True, scale_factor=args.img_scale_factor, s_j = standard_j) 
        test_dataset = make_hand_data_loader(
            args, args.val_yaml, False, is_train=False, scale_factor=args.img_scale_factor, s_j = standard_j) 

        if args.dataset == "both":
            o_dataset = CustomDataset_g(args, general_path + "/annotations/train", standard_j)
            train_dataset = ConcatDataset([train_dataset, o_dataset])
            
    elif args.dataset == "dart":
        train_dataset = DARTset(args, data_split='train')
        test_dataset = DARTset(args, data_split='test')

    elif args.dataset == "GAN":
        dataset = GAN(args)
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
        
    elif args.dataset == "SyntheticHands":
        dataset = SyntheticHands(args)
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
        
    else:
        train_path = os.path.join(general_path, "annotations/train")
        eval_path = os.path.join(general_path, "annotations/val")
        train_dataset = CustomDataset_g(args, train_path, standard_j)
        test_dataset = val_g_set(args, eval_path, standard_j)
        
    return train_dataset, test_dataset

    

class CustomDataset_g(Dataset):
    def __init__(self, args, path, standard_j):
        self.args = args
        self.noise_factor = 0.4
        self.path = path
        self.phase = path.split("/")[-1]
        self.root = "/".join(path.split("/")[:-2])
        with open(f"{path}/revision_data.pkl", "rb") as st_json:
            self.meta = pickle.load(st_json)
        self.bg_path = "../../datasets/without_bg_revision/background"
        self.bg_list = os.listdir(self.bg_path)
        self.s_j = standard_j
        self.img_path = os.path.join(self.root, f"images/{self.phase}")
        self.raw_res = 512
        self.img_res = 224
        self.scale_factor = 0.25
        self.rot_factor = 90 
        self.args.logger.debug('phase: {} => noise_factor: {}, scale_factor: {}, rot_factor: {}, raw_res: {}, img_res: {}'.format(self.phase, self.__dict__.get('noise_factor'), self.__dict__.get('scale_factor'),
                                                                                                                    self.__dict__.get('rot_factor'), self.__dict__.get('raw_res'), self.__dict__.get('img_res')))
        
    def __len__(self):
        return int(self.args.ratio_of_dataset * (len(self.meta) - 1))
        
    def __getitem__(self, idx):
        
        image, joint_2d, joint_3d = self.aug(idx)
        image = self.img_preprocessing(idx, image)

        transformed_img = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Resize((224, 224))])(image)
        
        return transformed_img, joint_2d, joint_3d
    
    def img_preprocessing(self, idx, rgb_img):
        # in the rgb image we add pixel noise in a channel-wise manner
        if self.phase == 'train':
            if idx < self.__len__():
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
        rgb_img = torch.from_numpy(rgb_img)
        return rgb_img

    def get_value(self):
        rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
        scale = min(1+self.scale_factor,
                max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
        
        return rot, scale
    
    def aug(self, idx):
        name = '/'.join(self.meta[idx]['file_name'].split('/')[1:])
        image = cv2.imread(os.path.join(self.img_path, name))  # PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        joint_2d = np.array(self.meta[idx]['joint_2d'])
        joint_3d = self.meta[idx]['joint_3d']    
        scale = self.meta[idx]['scale']   
        rot = self.meta[idx]['rot']       

        cropped_img = crop(image, (self.raw_res/2, self.raw_res/2), scale, [self.raw_res, self.raw_res], rot=rot)    
        bg_img = cv2.imread(os.path.join(self.bg_path, self.bg_list[idx%len(self.bg_list)]))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = cv2.resize(bg_img, (512, 512))
        
        if self.args.arm:
            new_image = np.zeros((512, 512, 3))
            
            index = np.where((cropped_img[:, :, 0] != 0) | (cropped_img[:, :, 1] != 0) | (cropped_img[:, :, 2] != 0))
            index = np.array(index)
            bbox_image = [min(index[0]), max(index[0]), min(index[1]), max(index[1])]
            bbox = [min(joint_2d[:, 1]),  max(joint_2d[:, 1]), min(joint_2d[:, 0]), max(joint_2d[:, 0])]
            dia_of_bbox = np.sqrt((bbox[0] - bbox[1]) ** 2 + (bbox[2] -bbox[3]) ** 2) / 512
            dia_of_hand = np.sqrt((bbox_image[0] - bbox_image[1]) ** 2 + (bbox_image[2] - bbox_image[3]) ** 2) / 512
            ratio_of_dia = dia_of_bbox / dia_of_hand
            a = int(max(bbox[0] - (ratio_of_dia * 100), 0))
            b = int(min(bbox[1] + (ratio_of_dia * 100), 512))
            c = int(max(bbox[2] - (ratio_of_dia * 100), 0))
            d = int(min(bbox[3] + (ratio_of_dia * 100), 512))
            new_image[a : b, c : d] = cropped_img[a : b, c: d]
            cropped_img = new_image.copy()
            
        index = np.where((cropped_img[:, :, 0] == 0) & (cropped_img[:, :, 1] == 0) & (cropped_img[:, :, 2] == 0))
        cropped_img[index] = bg_img[index]

        # if self.args.rot_j:
        #     joint_3d = self.j3d_processing(joint_3d, rot)  
        
        joint_2d = joint_2d / self.raw_res
        joint_2d, joint_3d = torch.tensor(joint_2d).float(), torch.tensor(joint_3d).float()
        
        image = cv2.resize(cropped_img, (224, 224))
        
        return image, joint_2d, joint_3d

    def j3d_processing(self, S, r):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S = np.einsum('ij,kj->ki', rot_mat, S) 
        S = S.astype('float32') 
        return S


        
class val_g_set(CustomDataset_g):
    def __init__(self,  *args):
        super().__init__(*args)
        with open(f"{self.path}/revision_data.pkl", "rb") as st_json:
            self.meta = pickle.load(st_json)
        self.img_path = os.path.join(self.root,f"images/{self.phase}" )

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

def myimresize(img, size, return_scale=False):

    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, (size[0],size[1]))
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

   
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

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, p):
        hms = np.zeros(shape=(self.num_parts, self.output_res,
                       self.output_res), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(p):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

    
        
