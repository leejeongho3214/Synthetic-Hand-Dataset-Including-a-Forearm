import sys
from tqdm import tqdm
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.miscellaneous import mkdir
from src.utils.comm import is_main_process
from src.datasets.build import make_hand_data_loader
import json
import math
import torch
from src.utils.dataset_loader import Coco, Dataset_interhand, HIU_Dataset, Panoptic, Rhd, GenerateHeatmap, add_our, our_cat
import os.path as op
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split

def build_dataset(args):
    assert args.name.split("/")[0] in ["simplebaseline", "hourglass", "hrnet", "ours"], "Your name of model is the wrong => %s" % args.name.split("/")[0]
    assert args.name.split("/")[1] in ["wrist", "general"] , "Your name of view is the wrong %s" % args.name.split("/")[1] 
    assert args.name.split("/")[2] in ["rhd", "coco", "frei", "panoptic", "hiu", "interhand", "ours"], "Your name of dataset is the wrong => %s" % args.name.split("/")[2]

    if "3d" in args.name.split("/")[3].split("_"): args.D3 = True
    args.dataset = args.name.split("/")[2]
    args.view = args.name.split("/")[1]
    args.model = args.name.split("/")[0]
    
    if args.eval:
        test_dataset = eval_set(args)
        return test_dataset, test_dataset
    
    path = "../../../../../../data1/ArmoHand/training"
    if not os.path.isdir(path):
        path = "../../datasets/ArmoHand/training"
        
    general_path = "../../../../../../data1/general_2M" # general-view image path (about 2M)
    if not os.path.isdir(general_path):
        general_path = "../../datasets/general_2M"
        
    if args.name.split("/")[1] == "wrist":
        folder = os.listdir(path)
        folder_num = [i for i in folder if i not in ["README.txt", "data.zip"]]
        
    if args.dataset == "interhand":
        dataset = Dataset_interhand(transforms.ToTensor(), "train", args)     
        trainset_dataset, test_dataset = add_our(args, dataset, folder_num, path)
        return trainset_dataset, testset_dataset

    if args.dataset  == "hiu":

        dataset = HIU_Dataset(args)
        trainset_dataset, test_dataset = add_our(args, dataset, folder_num, path)                 
        return trainset_dataset, testset_dataset

    if args.dataset == "panoptic":

        dataset = Panoptic(args)
        trainset_dataset, test_dataset = add_our(args, dataset, folder_num, path)                            
        return trainset_dataset, testset_dataset

    if args.dataset == "coco":

        dataset = Coco(args)
        trainset_dataset, test_dataset = add_our(args, dataset, folder_num, path)                    
        return trainset_dataset, testset_dataset

    if args.dataset == "frei":
        trainset_dataset = make_hand_data_loader(
            args, args.train_yaml, False, is_train=True, scale_factor=args.img_scale_factor) 
        testset_dataset = make_hand_data_loader(
            args, args.val_yaml, False, is_train=False, scale_factor=args.img_scale_factor)        
        return trainset_dataset, testset_dataset

    if args.dataset == "rhd":
        dataset = Rhd(args)
        trainset_dataset, test_dataset = add_our(args, dataset, folder_num, path)                 
        return trainset_dataset, testset_dataset
    
    else:
        if args.view == "wrist":
            eval_path = "/".join(path.split('/')[:-1]) + "/annotations/evaluation"
            test_dataset = val_set(args , 0, eval_path, args.color,
                                        args.ratio_of_aug, args.ratio_of_our)
            train_dataset = our_cat(args,folder_num, path)
        else:
            
            dataset = CustomDataset_g(args, general_path)
            train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

    return train_dataset, test_dataset



class CustomDataset(Dataset):
    def __init__(self, args, degree, path, color=False, ratio_of_aug=0.2, ratio_of_dataset=1):
        self.args = args
        self.color = color
        self.degree = degree
        self.path = path
        self.ratio_of_aug = ratio_of_aug
        self.ratio_of_dataset = ratio_of_dataset
        self.img_path = f'{path}/{degree}/images/train'
        try:
            if degree == None:
                with open(f"{path}/annotations/train/CISLAB_train_data_update.json", "r") as st_json:   ## When it has only one json file
                    self.meta = json.load(st_json)
                    self.img_path = f'{path}/images/train'
            else:
                with open(f"{path}/{degree}/annotations/train/CISLAB_train_data_update.json", "r") as st_json:
                    self.meta = json.load(st_json)              
        except:
            self.meta = None           ## When it calls eval_set inheritng parent's class
        
    
    def __len__(self):
        return int(len(self.meta['images']) * self.ratio_of_dataset)     

    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        move = self.meta['images'][idx]['move']
        degrees = self.meta['images'][idx]['degree']
        image = cv2.imread(os.path.join(self.img_path, name))   ## Color order of cv2 is BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.args.model == "ours":
            image_size = 256
        else:
            image_size = 224

        trans_option = {
            'resize': transforms.Resize((image_size, image_size)),
            'to_tensor': transforms.ToTensor(),
            'color': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            'norm': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        }
        
        if not self.color:
            del trans_option['color']
        
        image = i_rotate(image, degrees, 0, move)
        image = Image.fromarray(image)
        joint_2d = torch.tensor(self.meta['images'][idx]['rot_joint_2d'])
        
        if idx < len(self.meta['images']) * self.ratio_of_aug:

            trans = transforms.Compose([trans_option[i] for i in trans_option])

            
        else:
            trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        
        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d/4)
            
        joint_3d = torch.tensor(self.meta['images'][idx]['joint_3d'])


        return image, joint_2d, heatmap, joint_3d


class CustomDataset_g(Dataset):
    def __init__(self, args, path):
        with open(f"{path}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)
        with open(f"{path}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        self.args = args
        self.root = os.path.join(path, "images/train")
        
    def __len__(self):
        return len(self.meta['images'])
        
    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        id = self.meta['images'][idx]['frame_idx']
        image = cv2.imread(os.path.join(self.root, name))  # PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.args.model == "ours":
            image_size = 256
        else:
            image_size = 224
            
        image = Image.fromarray(image)

        trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]), p=self.args.ratio_of_aug),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ])
        
        image = trans(image)
            
        joint_3d = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
        joint_3d[:, 0] = - joint_3d[:, 0]
        joint_3d = joint_3d - joint_3d[0, :]

        return image, joint_3d, joint_3d, joint_3d
    
class val_set(CustomDataset):
    def __init__(self,  *args):
        super().__init__(*args)
        self.ratio_of_aug = 0
        self.ratio_of_dataset = 1
        with open(os.path.join(self.path, "evaluation_data_update.json"), "r") as st_json:
            self.meta = json.load(st_json)
        self.img_path = "/".join(self.path.split('/')[:-2]) +"/images/evaluation"

class eval_set(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_path = f'../../datasets/test/rgb'
        self.anno_path = f'../../datasets/test/annotations.json'
        self.list = os.listdir(self.image_path)

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        if not self.args.model == "ours":
            size = 256
        else:
            size = 224

        with open(self.anno_path, "r") as st_json:
            json_data = json.load(st_json)
            joint = json_data[f"{idx}"]['coordinates']
            pose_type = json_data[f"{idx}"]['pose_ctgy']
            file_name = json_data[f"{idx}"]['file_name']
            visible = json_data[f"{idx}"]['visible']
            try: 
                joint_2d = torch.tensor(joint)[:, :2]
            except:
                print(file_name)
                print("EROOORROORR")
            visible = torch.tensor(visible)
            joint_2d_v = torch.concat([joint_2d, visible[:, None]], axis = 1)
            assert len(joint) == 21, f"{file_name} have joint error"
            assert len(visible) == 21, f"{file_name} have visible error"
            
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        image = Image.open(f"../../datasets/{file_name}")
        trans_image = trans(image)
        joint_2d_v[:, 0] = joint_2d_v[:, 0] * image.width
        joint_2d_v[:, 1] = joint_2d_v[:, 1] * image.height
        joint_2d[:, 0] = joint_2d[:, 0] * image.width
        joint_2d[:, 1] = joint_2d[:, 1] * image.height
        
        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d / 2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)
            
        if self.args.eval:
            return trans_image, joint_2d_v, heatmap, pose_type
        else:
            return trans_image, joint_2d_v, heatmap, joint_2d_v


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




def apply(img, aug, num=1, scale=1.5):
    Y = [aug(img) for _ in range(num)]
    return Y


def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)
    result = cv2.warpAffine(result, translation, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)

    return result


class Json_transform(Dataset):
    def __init__(self, degree, path):
        self.degree = degree
        self.path = path
        self.degree = degree
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)
        self.root = f'{self.path}/{self.degree}/images/train'
        self.store_path = f'{self.path}/{self.degree}/annotations/train/CISLAB_train_data_update.json'

    def get_json(self):
        meta_list = self.meta['images'].copy()
        index = []
        pbar = tqdm(total = len(meta_list))
        for idx, j in enumerate(meta_list):
            pbar.update(1)
            if j['camera'] == '0':
                index.append(idx)
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(os.path.join(self.root, name))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            left_pixel, right_pixel = [79-112, -112], [174-112, -112]
            left_rot = math.cos(
                rad) * left_pixel[1] - math.sin(rad) * left_pixel[0] + 112
            right_rot = math.cos(
                rad) * right_pixel[1] - math.sin(rad) * right_pixel[0] + 112

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

                if i == 0:  # 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue

            d = joint_2d.clone()
            x = joint_2d[:, 0] - 112
            y = joint_2d[:, 1] - 112
            joint_2d[:, 0] = math.cos(rad) * x + math.sin(rad) * y + 112
            joint_2d[:, 1] = math.cos(
                rad) * y - math.sin(rad) * x + 112 + move_y + move_y2

            flag = False
            for o in joint_2d:
                if o[0] > 223 or o[1] > 223:
                    flag = True
                    index.append(idx)
                    break
            if flag:
                continue
                    
            rot_image = i_rotate(ori_image, degrees, 0, move_y + move_y2)
            rot_image = Image.fromarray(rot_image)
            
            j['joint_2d'] = d.tolist()
            j['joint_3d'] = joint.tolist()
            j['rot_joint_2d'] = joint_2d.tolist()
            j['degree'] = degrees
            j['move'] = move_y + move_y2
            j["rot_images"] = rot_image

        count = 0
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        with open(self.store_path, 'w') as f:
            json.dump(self.meta, f)

        print(
            f"Done ===> {self.store_path}")
        
        
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





class Our_testset_media(Dataset):
    def __init__(self, path, folder_name):

        self.image_path = f'{path}/{folder_name}/rgb'
        self.anno_path = f'{path}/{folder_name}/annotations'
        self.list = os.listdir(self.image_path)

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
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
                joint_2d.append(
                    [joint[f'{h}']['x'], joint[f'{h}']['y'], joint[f'{h}']['z']])
        joint_2d = torch.tensor(joint_2d)

        return image, joint_2d



    
class Json_e(Json_transform):
    def __init__(self, phase):
        root = "../../../../../../data1/ArmoHand"
        if phase == 'eval':
            try:
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)       
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
            except:
                root = "../../datasets/ArmoHand"
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)   
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
            
        else:
            root = "../../datasets/general_2M"
            with open(os.path.join(root, "annotations/train/CISLAB_train_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(root, "annotations/train/CISLAB_train_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(root, "annotations/train/CISLAB_train_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
                
            self.root = os.path.join(root, "images/train")
            self.store_path = os.path.join(root, "annotations/train/CISLAB_train_data_update.json")
    
# def main():
#     Json_e(phase = 'train').get_json()
#     print("ENDDDDDD")
    
# if __name__ == '__main__':
#     main()
    
        
