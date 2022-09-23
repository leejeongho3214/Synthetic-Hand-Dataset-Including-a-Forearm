import json
import os
import os.path as op
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir


class CustomDataset_train(Dataset):
    def __init__(self, args):
        self.num = int(args.train_data[9:-1])
        with open(f"../../datasets/our_data/CISLAB_HAND_{self.num}K/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"../../datasets/our_data/CISLAB_HAND_{self.num}K/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"../../datasets/our_data/CISLAB_HAND_{self.num}K/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)



    def __len__(self):
        return len(self.meta['images'])


    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        camera = self.meta['images'][idx]['camera']
        id = self.meta['images'][idx]['frame_idx']
        image = Image.open(f'../../datasets/our_data/CISLAB_HAND_{self.num}K/images/train/{name}')
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
        focal_length = self.camera['0']['focal'][f'{camera}'][0] * (6 / 7)  ## only one scalar (later u need x,y focal_length)
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

    def __len__(self):
        return len(self.anno)

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

class HIU_Dataset(Dataset):
    def __init__(self):
        image_list = []
        for (root, directories, files) in os.walk("../../datasets/HIU_DMTL"):
            for file in files:
                if not 'mask.png' in file:
                    if not 'mask.jpg' in file:
                        if not '.json' in file:
                            file_path = os.path.join(root, file)
                            anno_name = file_path[:-4] + '.json'
                            image_list.append((file_path, anno_name))
        self.image = image_list

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = Image.open(self.image[idx][0])
        scale = 224 / image.height
        with open(self.image[idx][1], "r") as st_json:
            annotation = json.load(st_json)
        if annotation['hand_type'][0] == 0:
            joint = annotation['pts2d_2hand'][21:]
        else:
            joint = annotation['pts2d_2hand'][:21]
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)[(2, 1, 0), :, :]
        c = torch.tensor(joint) * scale

        return trans_image, c

class Our_testset(Dataset):
    def __init__(self):
        self.image_path = '../../datasets/our_testset/rgb'
        self.anno_path = '../../datasets/our_testset/annotation'
    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path,f'{idx}.jpg'))
        scale_x = 224 / image.width
        scale_y = 224 / image.height
        with open(os.path.join(self.anno_path,f'{idx}.json'), "r") as st_json:
            json_data = json.load(st_json)
            joint_total = json_data['annotations']
            joint = {}
            joint_2d = []
            for j in joint_total:
                if j['label'] != 'Pose':
                    if len(j['metadata']['system']['attributes']) > 0:
                        # Change 'z' to 'indicator function'
                        # Ex. 0 means visible joint, 1 means invisible joint
                        j['coordinates']['z'] = 0
                        joint[f"{int(j['label'])}"] = j['coordinates']
                    else:
                        j['coordinates']['z'] = 1
                        joint[f"{int(j['label'])}"] = j['coordinates']

            if len(joint) < 21:
                assert f"This {idx}.json is not correct"

            for h in range(0, 21):
                joint_2d.append([joint[f'{h}']['x'], joint[f'{h}']['y'], joint[f'{h}']['z']])

        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_image = trans(image)[(2, 1, 0), :, :]
        c = torch.tensor(joint_2d)
        c[:,0] = c[:,0] * scale_x
        c[:, 1] = c[:, 1] * scale_y

        return trans_image, c