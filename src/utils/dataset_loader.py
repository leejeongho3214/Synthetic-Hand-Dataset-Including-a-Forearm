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
 