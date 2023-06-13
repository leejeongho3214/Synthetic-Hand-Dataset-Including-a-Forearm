import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

class GAN(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/GANeratedHands_Release/data/noObject"
        self.img_list = list()
        for (root, _, files) in os.walk(self.img_path):
            for file in files:
                if '.png' in file:
                    self.img_list.append(os.path.join(root , file))

            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):

        size = 224
        root = '/'.join(self.img_list[idx].split('/')[:7])
        file_index = self.img_list[idx].split('/')[7].split('_')[0]
        image = Image.open(self.img_list[idx])
        anno_3d_txt = open(os.path.join(root, (file_index + "_joint_pos_global.txt")))
        line_3d = anno_3d_txt.readline()
        anno_3d = list(map(float, line_3d.strip().split(',')))
        anno_3d = np.reshape(anno_3d, (21, 3)) / 10000
        
        anno_2d_txt = open(os.path.join(root, (file_index + "_joint2D.txt")))
        line_2d = anno_2d_txt.readline()
        anno_2d = list(map(float, line_2d.strip().split(',')))
        anno_2d = np.reshape(anno_2d, (21, 2)) / 256

        anno_2d, anno_3d = torch.tensor(anno_2d, dtype = torch.float32), torch.tensor(anno_3d, dtype = torch.float32)
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)

        return trans_image, anno_2d, anno_3d
    
import os
import json
from PIL import Image
from torchvision import transforms
import torch

class Frei(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/frei_test/evaluation/rgb"
        with open("../../datasets/frei_test/evaluation_K.json", "r") as st_json:
            self.anno_K = json.load(st_json)
        with open("../../datasets/frei_test/evaluation_xyz.json", "r") as st_json:
            self.anno_xyz = json.load(st_json)
        # with open("../../datasets/frei_test/evaluation_mano.json", "r") as st_json:
        #     self.anno_mano = json.load(st_json)
            
    def __len__(self):
        return len(self.anno_K)
    
    def __getitem__(self, idx):
        anno_K = torch.tensor(self.anno_K[idx])
        anno_xyz = torch.tensor(self.anno_xyz[idx])
        # anno_mano = torch.tensor(self.anno_mano[idx][0][:-3])
        
        joint_2d = torch.matmul(anno_K, anno_xyz.T).T
        joint_2d = (joint_2d[:, :2].T / joint_2d[:, -1]).T
        
        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        
        image = Image.open(os.path.join(self.img_path, f"{str(idx).zfill(8)}.jpg"))
        scale_x = size / image.width
        scale_y = size / image.height

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y


        return trans_image, joint_2d, anno_xyz
 