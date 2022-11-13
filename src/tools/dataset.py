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
from d2l import torch as d2l
import sys
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir
import torchvision

def apply(img, aug, num=1, scale=1.5):
    Y = [aug(img) for _ in range(num)]
    return Y


def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    translation = np.float32([[1,0,move_x], [0,1,move_y]])
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h), flags = cv2.INTER_LINEAR, borderMode = cv2.INTER_LINEAR)
    result = cv2.warpAffine(result, translation, (new_w, new_h), flags = cv2.INTER_LINEAR, borderMode = cv2.INTER_LINEAR)
    
    return result

class Json_transform(Dataset):
    def __init__(self, degree, path, rotation = False, color = False):
        self.degree = degree
        self.rotation = rotation
        self.color = color
        self.path = path
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
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

            ori_image[ori_image < 30] = 0 ## remove the noise, not hand pixel

            root = "../../datasets/background/bg"
            path1 = os.listdir(root)
            bg = cv2.imread(os.path.join(root, random.choice(path1)))
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg = cv2.resize(bg, (224,224))

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            left_pixel, right_pixel = [79-112, -112], [174-112, -112]
            left_rot = math.cos(rad) * left_pixel[1] - math.sin(rad) * left_pixel[0] + 112
            right_rot = math.cos(rad) * right_pixel[1] - math.sin(rad) * right_pixel[0] + 112

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
                    
                if i == 0:  ## 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue
            
            d = joint_2d.clone()
            x = joint_2d[:,0] - 112
            y = joint_2d[:,1] - 112
            joint_2d[:,0] =  math.cos(rad) * x + math.sin(rad) * y + 112
            joint_2d[:,1] = math.cos(rad) * y - math.sin(rad) * x + 112 + move_y + move_y2

            flag = False
            for o in joint_2d:
                if o[0] > 223 or o[1] > 223:
                    flag =True
                    index.append(idx)
                    break
            if flag:
                continue
            j['joint_2d'] = d.tolist()
            j['joint_3d'] = joint.tolist()
            j['rot_joint_2d'] = joint_2d.tolist()
            j['degree'] = degrees
            j['move'] = move_y + move_y2
            
        count = 0 
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data_update.json", 'w') as f:
            json.dump(self.meta, f)

        print(f"Done ===> {path}/{degree}/annotations/train/CISLAB_train_data_update.json")
        # assert False, "finish"

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):
        

        trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = np.array(self.meta['images'][idx]['image'])
        image = trans(image)
        joint_2d = self.meta['images'][idx]['joint_2d']
        joint_3d = self.meta['images'][idx]['joint_3d']

        return image, joint_2d, joint_3d

class CustomDataset_train_new(Dataset):
    def __init__(self, degree, path, rotation = False, color = False, blur = False, erase = False, ratio = 0.2):
        self.rotation =rotation
        self.color = color
        self.degree = degree
        self.path = path
        self.ratio = ratio
        self.blur = blur
        self.erase = erase
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data_update.json", "r") as st_json:
            self.meta = json.load(st_json)

    def __len__(self):
        return len(self.meta['images'])
    

    def __getitem__(self, idx):

        name = self.meta['images'][idx]['file_name']
        move = self.meta['images'][idx]['move']
        degrees = self.meta['images'][idx]['degree']
        image = cv2.imread(f'{self.path}/{self.degree}/images/train/{name}') ## PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if idx < len(self.meta['images']) * self.ratio: 

            if self.rotation:
                image  = i_rotate(image, degrees, 0, move)
                image = Image.fromarray(image)
                joint_2d = torch.tensor(self.meta['images'][idx]['rot_joint_2d']) 

            else:
                image = Image.fromarray(image)
                joint_2d = torch.tensor(self.meta['images'][idx]['joint_2d'])
        else:
            image = Image.fromarray(image)
            joint_2d = torch.tensor(self.meta['images'][idx]['joint_2d'])

        trans_option = {
                        'resize':transforms.Resize((224, 224)),
                        'to_tensor':transforms.ToTensor(),
                        'color':transforms.ColorJitter(brightness= 0.5, contrast= 0.5, saturation= 0.5, hue= 0.5),
                        'blur':transforms.GaussianBlur(kernel_size= (5, 5), sigma=(0.1, 1)),
                        'erase':transforms.RandomErasing(p=1.0, scale=(0.02, 0.04), ratio = (0.3, 3.3)),
                        'norm':transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        }

        if not self.blur: del trans_option['blur']
        if not self.color: del trans_option['color']
        if not self.erase: del trans_option['erase']

        trans = transforms.Compose([trans_option[i] for i in trans_option])                      
        image = trans(image)

        joint_3d = torch.tensor(self.meta['images'][idx]['joint_3d'])


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

def save_checkpoint(model, args, epoch,optimizer, best_loss,count, ment, num_trial=10, iteration = 0, logger=None):
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
                'count': count,
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
        joint_2d = torch.tensor(joint)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        return trans_image, joint_2d, joint_2d



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
        joint_2d = torch.tensor(joint)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y
        
        import math
        def cal_rad(arr):
            rad = math.atan2(arr[3]-arr[1],arr[2]-arr[0])

            return rad


        point = [0,0, joint_2d[0,0]-112, joint_2d[0,1]-112]
        rad = cal_rad(point)
        import cv2
        degree = math.degrees(rad) + 270 

        iimage = cv2.imread(self.image[idx][0])
        iimage = cv2.resize(iimage, (224,224))
        result, matrix = im_rotate(iimage, degree)

        x = joint_2d[:,0] - 112
        y = joint_2d[:,1] - 112
        rad = math.radians(degree)
        joint_2d[:,0] =  math.cos(rad) * x + math.sin(rad) * y + 112
        joint_2d[:,1] = math.cos(rad) * y - math.sin(rad) * x + 112

        # visualize(result, joint_2d)
        pil_img = Image.fromarray(result)
        trans_image = trans(pil_img)
        joint_2d = torch.tensor(joint_2d).detach().clone()
        z = torch.zeros(21,1)
        joint_3d = torch.concat([joint_2d, z], 1)

        return trans_image, joint_2d, joint_2d


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
        joint_2d = torch.tensor(joint_2d)
        joint_2d[:,0] = joint_2d[:,0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        return trans_image, joint_2d