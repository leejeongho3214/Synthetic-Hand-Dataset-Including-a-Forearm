import json

import os.path as op
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir


class CustomDataset_train(Dataset):
    def __init__(self,args):
        if int(args.train_data[-3]) > 0:
            with open(f"../../datasets/our_data/CISLAB_HAND_{int(args.train_data[-3])}0K/annotations/train/CISLAB_train_camera.json", "r") as st_json:
                self.camera = json.load(st_json)
            with open(f"../../datasets/our_data/CISLAB_HAND_{int(args.train_data[-3])}0K/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
                self.joint = json.load(st_json)
            with open(f"../../datasets/our_data/CISLAB_HAND_{int(args.train_data[-3])}0K/annotations/train/CISLAB_train_data.json", "r") as st_json:
                self.meta = json.load(st_json)
            self.num = int(args.train_data[-3])
        else:
            assert 'It is not correct dataset name'

    # 총 데이터의 개수를 리턴
    def __len__(self):
        # return len(self.meta['images'])
        return 5000

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        camera = self.meta['images'][idx]['camera']
        id = self.meta['images'][idx]['frame_idx']
        image = Image.open(f'../../datasets/our_data/CISLAB_HAND_{self.num}0K/images/train/{name}')
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        if id > 11499:
            joint = torch.tensor(self.joint['1'][f'{id}']['world_coord'][21:])
            focal_length = self.camera['1']['focal'][f'{camera}'][0] * (
                    6 / 7)  ## only one scalar (later u need x,y focal_length)
            translation = self.camera['1']['campos'][f'{camera}']
            rot = self.camera['1']['camrot'][f'{camera}']
        else:
            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0] * (
                    6 / 7)  ## only one scalar (later u need x,y focal_length)
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']

        c = []

        for i in range(21):
            a = np.dot(np.array(rot, dtype='float32'),
                       np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
            a[:2] = a[:2] / a[2]
            b = a[:2] * focal_length + 112
            b = torch.tensor(b)
            if i == 0:  ## 112 is image center
                c = b
            elif i == 1:
                c = torch.stack([c, b], dim=0)
            else:
                c = torch.concat([c, b.reshape(1, 2)], dim=0)


        return image,  c, joint


class CustomDataset_test(Dataset):
    def __init__(self):
        with open("../../datasets/our_data/Test_Set/annotations.json", "r") as st_json:
            self.anno = json.load(st_json)

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.anno)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        name = self.anno[idx]['file_name'][7:]
        image = Image.open(f'../../datasets/our_data/Test_Set/ori_images/{name}')
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_image = trans(image)[(2,1,0),:,:]
        c = torch.tensor(self.anno[idx]['joint_2d'])

        return trans_image, c, c


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

def save_checkpoint(model, args, epoch,optimizer, best_loss,ment, num_trial=10, iteration = 0, logger=None):
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
                'iteration': iteration,
                'optimizer_state_dict': optimizer,
                'best_loss': best_loss,
                'model_state_dict': model_to_save.state_dict()}, op.join(checkpoint_dir, 'state_dict.bin'))
            logger.info("Save checkpoint to epoch:{}_iter:{}_{}".format(epoch, iteration, checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return model_to_save, checkpoint_dir