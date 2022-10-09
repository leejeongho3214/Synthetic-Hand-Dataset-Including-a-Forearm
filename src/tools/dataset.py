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

class CustomDataset_train_new(Dataset):
    def __init__(self, degree):
        self.degree = degree
        # self.num = int(args.train_data[9:-1])
        with open(f"../../datasets/CISLAB_various/{degree}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"../../datasets/CISLAB_various/{degree}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"../../datasets/CISLAB_various/{degree}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)

        index = []
        for idx, j in enumerate(self.meta['images']):
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
            for i in range(21):
                a = np.dot(np.array(rot, dtype='float32'),
                        np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
                a[:2] = a[:2] / a[2]
                b = a[:2] * focal_length + 112
                b = torch.tensor(b)
                for o in b:
                    if o >223:
                        flag = True
                        index.append(idx)
                        break
                if flag:
                    break
            if flag:
                continue
        count = 0 
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        
        



    def __len__(self):
        return len(self.meta['images'])


    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        camera = self.meta['images'][idx]['camera']
        id = self.meta['images'][idx]['frame_idx']
        image = Image.open(f'../../datasets/CISLAB_various/{self.degree}/images/train/{name}')
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
        focal_length = self.camera['0']['focal'][f'{camera}'][0]  ## only one scalar (later u need x,y focal_length)
        translation = self.camera['0']['campos'][f'{camera}']
        rot = self.camera['0']['camrot'][f'{camera}']
        # rot2 = sum(rot, [])
        # parameter =[]
        # parameter.append(focal_length)
        # for i in rot2:
        #     parameter.append(i)
        # for i in translation:
        #     parameter.append(i)
        # camera_parameter = torch.tensor(parameter)

        c = []

        for i in range(21):
            a = np.dot(np.array(rot, dtype='float32'),
                       np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
            a[:2] = a[:2] / a[2]
            b = a[:2] * focal_length + 112
            b = torch.tensor(b)
            for o in b:
                assert o <223, "joint is out of images"

            if i == 0:  ## 112 is image center
                c = b
            elif i == 1:
                c = torch.stack([c, b], dim=0)
            else:
                c = torch.concat([c, b.reshape(1, 2)], dim=0)



        return image,  c, joint



class CustomDataset_train(Dataset):
    def __init__(self):
        # self.num = int(args.train_data[9:-1])
        with open(f"../../datasets/CISLAB_Full/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"../../datasets/CISLAB_Full/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"../../datasets/CISLAB_Full/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)
        for idx, i in enumerate(self.meta['images']):
            if i['camera'] == '0':
                del self.meta['images'][idx]



    def __len__(self):
        return len(self.meta['images'])


    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        camera = self.meta['images'][idx]['camera']
        # if camera == '0':
        #     break
        id = self.meta['images'][idx]['frame_idx']
        image = Image.open(f'../../datasets/CISLAB_Full/images/train/{name}')
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
        focal_length = self.camera['0']['focal'][f'{camera}'][0]  ## only one scalar (later u need x,y focal_length)
        translation = self.camera['0']['campos'][f'{camera}']
        rot = self.camera['0']['camrot'][f'{camera}']
        rot2 = sum(rot, [])
        parameter =[]
        parameter.append(focal_length)
        for i in rot2:
            parameter.append(i)
        for i in translation:
            parameter.append(i)
        camera_parameter = torch.tensor(parameter)

        c = []

        for i in range(21):
            a = np.dot(np.array(rot, dtype='float32'),
                       np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
            a[:2] = a[:2] / a[2]
            b = a[:2] * focal_length + 112
            b = torch.tensor(b)
            for o in b:
                if o >224:
                    assert "incorrect_image"
            if i == 0:  ## 112 is image center
                c = b
            elif i == 1:
                c = torch.stack([c, b], dim=0)
            else:
                c = torch.concat([c, b.reshape(1, 2)], dim=0)



        return image,  c, joint, camera_parameter

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

        return trans_image, c

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

        trans_image = trans(image)[(2, 1, 0), :, :]
        c = torch.tensor(joint)
        c[:, 0] = c[:, 0] * scale_x
        c[:, 1] = c[:, 1] * scale_y

        return trans_image, c



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

        trans_image = trans(image)[(2, 1, 0), :, :]
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


        return trans_image, torch.tensor(c)


class Our_testset(Dataset):
    def __init__(self, folder_name):
       
        self.image_path = f'../../datasets/our_testset/{folder_name}/rgb'
        self.anno_path = f'../../datasets/our_testset/{folder_name}/annotation'
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