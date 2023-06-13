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