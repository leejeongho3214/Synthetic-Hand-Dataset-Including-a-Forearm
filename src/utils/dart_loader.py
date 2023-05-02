import os
import pickle
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pytorch3d')))
import cv2
import imageio
import numpy as np
import torch
from torchvision import transforms
from pytorch3d.io import load_obj


RAW_IMAGE_SIZE = 512
BG_IMAGE_SIZE = 384
DATA_ROOT = "../../datasets/"

class DARTset():

    def __init__(self, args, data_split="train", use_full_wrist=True, load_wo_background=False):

        self.args = args
        self.rot_factor = 90
        self.scale_factor = 0.25
        self.noise_factor = 0.4
        self.name = "DARTset"
        self.data_split = data_split
        self.root = os.path.join(DATA_ROOT, self.name, self.data_split)
        self.load_wo_background = load_wo_background
        self.raw_img_size = RAW_IMAGE_SIZE
        self.img_size = RAW_IMAGE_SIZE if load_wo_background else BG_IMAGE_SIZE
        self.img_res = 224
        self.use_full_wrist = use_full_wrist

        obj_filename = os.path.join('../../datasets/assets/hand_mesh.obj')
        _, faces, _ = load_obj(
            obj_filename,
            device="cpu",
            load_textures=False,
        )
        self.reorder_idx = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
        self.hand_faces = faces[0].numpy()
        self.transform_func = transforms.Compose([
            transforms.Resize((224, 224)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.load_dataset()

    def load_dataset(self):

        self.image_paths = []
        self.raw_mano_param = []
        self.joints_3d = []
        self.verts_3d_paths = []
        self.joints_2d = []

        image_parts = [
            r for r in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, r)) and "verts" not in r and "wbg" not in r
        ]
        image_parts = sorted(image_parts)

        for imgs_dir in image_parts:
            imgs_path = os.path.join(self.root, imgs_dir)
            data_record = pickle.load(open(os.path.join(self.root, f"part_{imgs_dir}.pkl"), "rb"))
            for k in range(len(data_record["pose"])):
                self.image_paths.append(os.path.join(imgs_path, data_record["img"][k]))
                self.raw_mano_param.append(data_record["pose"][k].astype(np.float32))
                self.joints_3d.append(data_record["joint3d"][k].astype(np.float32))
                self.joints_2d.append(data_record["joint2d"][k].astype(np.float32))
                verts_3d_path = os.path.join(imgs_path + "_verts", data_record["img"][k].replace(".png", ".pkl"))
                self.verts_3d_paths.append(verts_3d_path)

        self.sample_idxs = list(range(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from src.tools.dataset import crop
        from matplotlib import pyplot as plt

        img = self.get_image(idx)
        scale_factor = 0.2
        joint, joint_3d = self.get_joints_2d(idx), self.get_joints_3d(idx)      ## get a joint location of image that is 224 x 224
        
        ''' Use a augmentation in only train phase
            And image resolution is 512 x 512 so don't confuse resolution before image resizing'''
        if self.data_split == "train":
            if idx < int(self.__len__() * 0.6):
                scale = min(1.2 + scale_factor, max(1.3 - scale_factor, np.random.randn()* scale_factor + 1.1))
                rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
                joint = torch.tensor(self.j2d_processing(np.array(joint), scale, rot))
                while not ((0 < joint.all()) and (joint.all() < 224)):
                    scale = min(1.2 + scale_factor, max(1.3 - scale_factor, np.random.randn()* scale_factor + 1.1))
                    rot = min(2*self.rot_factor,
                        max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
                    joint = torch.tensor(self.j2d_processing(np.array(joint), scale, rot))
                img = crop(img, [img.shape[1]/2, img.shape[1]/2], scale, [img.shape[1], img.shape[1]], rot = rot)
                # if self.args.rot_j: joint_3d = torch.tensor(self.j3d_processing(self.get_joints_3d(idx), rot))
            
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = self.transform_func(img)
        heatmap = GenerateHeatmap(56, 21)(joint / 4)
        
        # img = transforms.Resize([224, 224])(img)
        # ori_img = np.array(img.permute(1, 2, 0)).copy()

        # parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])    
        # for i in range(21):
        #     cv2.circle(ori_img, (int(joint[i][0]), int(joint[i][1])), 2, [0, 1, 0],
        #             thickness=-1)
        #     if i != 0:
        #         cv2.line(ori_img, (int(joint[i][0]), int(joint[i][1])),
        #                 (int(joint[parents[i]][0]), int(joint[parents[i]][1])),
        #                 [0, 0, 1], 1)
        # plt.imshow(ori_img)
        # plt.savefig("rot.jpg")

        return img, joint, joint_3d
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
    def get_joints_3d(self, idx):
        joints = self.joints_3d[idx].copy()
        # * Transfer from UNITY coordinate system
        joints[:, 1:] = -joints[:, 1:]
        joints = joints[self.reorder_idx]
        joints = joints - joints[9] + np.array([0, 0, 0.5])  # * We use ortho projection, so we need to shift the center of the hand to the origin
        # joints = joints - joints[0, :][None, :].repeat(21, axis = 0)
        joints = torch.from_numpy(joints).float()
        return joints

    def get_joints_2d(self, idx):
        joints_2d = self.joints_2d[idx].copy()[self.reorder_idx]
        # joints_2d = joints_2d / self.raw_img_size * 224
        joints_2d = joints_2d / self.raw_img_size * self.img_res

        joints_2d = torch.from_numpy(joints_2d).float()
        # joints_2d = joints_2d / self.raw_img_size * self.img_size
        return joints_2d

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_image(self, idx):
        path = self.image_paths[idx]
        if self.load_wo_background:
            img = np.array(imageio.imread(path, pilmode="RGBA"), dtype=np.uint8)
            img = img[:, :, :3]
        else:
            path = os.path.join(*path.split("/")[:-2], path.split("/")[-2] + "_wbg", path.split("/")[-1])
            img = cv2.imread(path)[..., ::-1]
            
        img = self.img_preprocessing(idx, img)

        return img
    
    def j2d_processing(self, kp, scale, r):
        from src.tools.dataset import transform
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, (self.img_res/2, self.img_res/2), scale, 
                                    [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        # kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # kp = kp.astype('float32')
        return kp

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
    
    def img_preprocessing(self, idx, rgb_img):

        # in the rgb image we add pixel noise in a channel-wise manner
        if self.data_split == 'train':
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
        # rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        rgb_img = rgb_img.astype('float32')/255.0
        
        return rgb_img

    def get_sides(self, idx):
        return "right"