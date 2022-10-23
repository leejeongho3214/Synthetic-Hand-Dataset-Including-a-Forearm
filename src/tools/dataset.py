import json
import math
import os
import os.path as op
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir


def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    translation = np.float32([[1,0,move_x], [0,1,move_y]])
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    result = cv2.warpAffine(result, translation, (new_w, new_h))
    
    return result

class CustomDataset_train_test(Dataset):
    def __init__(self):
        with open(f"{path}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"{path}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"{path}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)

        index = []
        for idx, j in enumerate(self.meta['images']):
                
            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]  ## only one scalar (later u need x,y focal_length)
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            image = cv2.imread(f'{path}/images/train/{name}')

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
                    
                if i == 0:  ## 112 is image center
                    c = b
                elif i == 1:
                    c = torch.stack([c, b], dim=0)
                else:
                    c = torch.concat([c, b.reshape(1, 2)], dim=0)
            if flag:
                continue

            j['image'] = image
            j['joint_2d'] = c
            j['joint_3d'] = joint
        
        count = 0 
        for w in index:
            del self.meta['images'][w-count]
            count += 1

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):

        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image = self.meta['images'][idx]['image']
        image = Image.fromarray(image)
        image = trans(image)
        joint_2d = self.meta['images'][idx]['joint_2d']
        joint_3d = self.meta['images'][idx]['joint_3d']

        return image,  joint_2d, joint_3d

class CustomDataset_train_new(Dataset):
    def __init__(self, degree, path):
        self.degree = degree
        # self.num = int(args.train_data[9:-1])
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)

        meta_list = self.meta['images'].copy()
        index = []
        for idx, j in enumerate(meta_list):
            if j['camera'] == '0':
                index.append(idx)
                continue
                
            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]  ## only one scalar (later u need x,y focal_length)
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(f'{path}/{self.degree}/images/train/{name}')
            degrees = random.uniform(-30, 30)
            move_x = random.uniform(-30, 30)
            move_y = random.uniform(0, 50)
            image  = i_rotate(ori_image, degrees, move_x, move_y)
            rad = math.radians(degrees)

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
                    
                if i == 0:  ## 112 is image center
                    c = b
                elif i == 1:
                    c = torch.stack([c, b], dim=0)
                else:
                    c = torch.concat([c, b.reshape(1, 2)], dim=0)
            if flag:
                continue
            
            d = c.clone()
            x = c[:,0] - 112
            y = c[:,1] - 112
            c[:,0] =  math.cos(rad) * x + math.sin(rad) * y + 112
            c[:,1] = math.cos(rad) * y - math.sin(rad) * x + 112 + move_y

            flag = False
            for o in c:
                if o[0] > 223 or o[1] > 223:
                    flag =True
                    index.append(idx)
                    break
            if flag:
                continue

            j['image'] = image
            j['joint_2d'] = c
            j['joint_3d'] = joint

            self.meta['images'].append({'image': ori_image, 'joint_2d': d, 'joint_3d':joint})
 
        count = 0 
        for w in index:
            del self.meta['images'][w-count]
            count += 1

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):

        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image = self.meta['images'][idx]['image']
        image = Image.fromarray(image)
        image = trans(image)
        joint_2d = self.meta['images'][idx]['joint_2d']
        joint_3d = self.meta['images'][idx]['joint_3d']

        return image, joint_2d, joint_3d


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
                if not '.json' in file:
                    if not '.DS_Store' in file:
                        file_path = os.path.join(root, file)
                        anno_name = file_path[:-4] + '.json'
                        image_list.append((file_path, anno_name))
        self.image = image_list

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = Image.open(self.image[idx][0])
        scale_x = 224 / image.width
        scale_y = 224 / image.height
        with open(self.image[idx][1], "r") as st_json:
            annotation = json.load(st_json)
        if annotation['hand_type'][0] == 0:
            joint = annotation['pts2d_2hand'][21:]
        else:
            joint = annotation['pts2d_2hand'][:21]
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        c = torch.tensor(joint)
        c[:, 0] = c[:, 0] * scale_x
        c[:, 1] = c[:, 1] * scale_y

        return trans_image, c, c



class HIU_Dataset_align(Dataset):
    def __init__(self):
        image_list = []
        for (root, directories, files) in os.walk("../../datasets/HIU_DMTL"):
            for file in files:
                if not '.json' in file:
                    if not '.DS_Store' in file:
                        file_path = os.path.join(root, file)
                        anno_name = file_path[:-4] + '.json'
                        image_list.append((file_path, anno_name))
        self.image = image_list

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        def im_rotate(img, degree):
            h, w = img.shape[:-1]

            crossLine = int(((w * h + h * w) ** 0.5))
            centerRotatePT = 112, 112
            new_h, new_w = 224, 224

            rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
            result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
            
            return result , rotatefigure


        from PIL import Image
        image = Image.open(self.image[idx][0])
        scale_x = 224 / image.width
        scale_y = 224 / image.height
        with open(self.image[idx][1], "r") as st_json:
            annotation = json.load(st_json)
        if annotation['hand_type'][0] == 0:
            joint = annotation['pts2d_2hand'][21:]
        else:
            joint = annotation['pts2d_2hand'][:21]
        from torchvision.transforms import transforms
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        c = torch.tensor(joint)
        c[:, 0] = c[:, 0] * scale_x
        c[:, 1] = c[:, 1] * scale_y
        
        import math
        def cal_rad(arr):
            rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])

            return rad


        point = [0,0, c[0,0]-112, c[0,1]-112]
        rad = cal_rad(point)
        import cv2
        degree = math.degrees(rad) + 270 

        iimage = cv2.imread(self.image[idx][0])
        iimage = cv2.resize(iimage, (224,224))
        result, matrix = im_rotate(iimage, degree)

        x = c[:,0] - 112
        y = c[:,1] - 112
        rad = math.radians(degree)
        c[:,0] =  math.cos(rad) * x + math.sin(rad) * y + 112
        c[:,1] = math.cos(rad) * y - math.sin(rad) * x + 112

        # visualize(result, c)
        pil_img = Image.fromarray(result)
        trans_image = trans(pil_img)
        joint_2d = torch.tensor(c).detach().clone()
        z = torch.zeros(21,1)
        joint_3d = torch.concat([joint_2d, z], 1)

        return trans_image, joint_2d, joint_2d

class Our_testset_media(Dataset):
    def __init__(self, path, folder_name):
       
        self.image_path = f'{path}/{folder_name}/rgb'
        self.anno_path = f'{path}/{folder_name}/annotations'
        self.list = os.listdir(self.image_path)

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        images = Image.open(os.path.join(self.image_path, self.list[idx]))
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_image = trans(images)[(2, 1, 0), :, :]
        image = cv2.imread(os.path.join(self.image_path, self.list[idx]))
        with open(os.path.join(self.anno_path, self.list[idx])[:-3]+"json", "r") as st_json:
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

        c = torch.tensor(joint_2d)
        return image, c, trans_image

class Our_testset(Dataset):
    def __init__(self, path, folder_name):
       
        self.image_path = f'{path}/{folder_name}/rgb'
        self.anno_path = f'{path}/{folder_name}/annotations'
        self.list = os.listdir(self.image_path)

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.list[idx]))
        scale_x = 224 / image.width
        scale_y = 224 / image.height
        with open(os.path.join(self.anno_path, self.list[idx])[:-3]+"json", "r") as st_json:
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
                                    
        trans_image = trans(image)
        c = torch.tensor(joint_2d)
        c[:,0] = c[:,0] * scale_x
        c[:, 1] = c[:, 1] * scale_y

        return trans_image, c