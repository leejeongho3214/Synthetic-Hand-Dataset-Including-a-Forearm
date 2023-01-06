import sys
from pycocotools.coco import COCO
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from src.utils.preprocessing import load_skeleton, process_bbox
from src.utils.miscellaneous import mkdir
from src.utils.comm import is_main_process
from src.datasets.build import make_hand_data_loader
from src.utils.transforms import world2cam, cam2pixel
import json
import math
import os
import os.path as op
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split, ConcatDataset
import torch


def build_dataset(args):

    if args.eval:
        test_dataset = Our_testset_new(args)
        return test_dataset, test_dataset
   
    path = "../../../../../../data1/1231"
    if not os.path.isdir(path):
        path = "../../datasets/synthetic_wrist"  # wrist-view image path (about 37K)
    general_path = "../../datasets/synthetic_general" # general-view image path (about 80K)
    folder = os.listdir(path)
    folder_num = [i for i in folder if i not in ["README.txt", "data.zip"]]
        
    if args.dataset == "interhand":

        dataset = Dataset_interhand(transforms.ToTensor(), "train", args)
        trainset_dataset, testset_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        
        for iter, degree in enumerate(folder_num):
                ratio  = ((len(trainset_dataset) + len(testset_dataset)) * args.ratio_of_other) / 373184
                dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

                if iter == 0:
                    train_dataset, test_dataset = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

                else:
                    train_dataset_other, test_dataset_other = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])
                    test_dataset = ConcatDataset(
                        [test_dataset, test_dataset_other])
                    
        trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
        testset_dataset = ConcatDataset([test_dataset, testset_dataset])
                    
        return trainset_dataset, testset_dataset

    if args.dataset == "hiu":

        dataset = HIU_Dataset(args)
        trainset_dataset, testset_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        
        for iter, degree in enumerate(folder_num):
                ratio  = ((len(trainset_dataset) + len(testset_dataset)) * args.ratio_of_other) / 373184
                dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

                if iter == 0:
                    train_dataset, test_dataset = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

                else:
                    train_dataset_other, test_dataset_other = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])
                    test_dataset = ConcatDataset(
                        [test_dataset, test_dataset_other])
                    
        trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
        testset_dataset = ConcatDataset([test_dataset, testset_dataset])
                    
        return trainset_dataset, testset_dataset

    if args.dataset == "panoptic":

        dataset = Panoptic(args)
        trainset_dataset, testset_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        
        for iter, degree in enumerate(folder_num):
                ratio  = ((len(trainset_dataset) + len(testset_dataset)) * args.ratio_of_other) / 373184
                dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

                if iter == 0:
                    train_dataset, test_dataset = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

                else:
                    train_dataset_other, test_dataset_other = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])
                    test_dataset = ConcatDataset(
                        [test_dataset, test_dataset_other])
                    
        trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
        testset_dataset = ConcatDataset([test_dataset, testset_dataset])
                    
        return trainset_dataset, testset_dataset

    if args.dataset == "coco":

        dataset = Coco(args)
        trainset_dataset, testset_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        
        for iter, degree in enumerate(folder_num):
                ratio  = ((len(trainset_dataset) + len(testset_dataset)) *args.ratio_of_other) / 373184
                dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

                if iter == 0:
                    train_dataset, test_dataset = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

                else:
                    train_dataset_other, test_dataset_other = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])
                    test_dataset = ConcatDataset(
                        [test_dataset, test_dataset_other])
                    
        trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
        testset_dataset = ConcatDataset([test_dataset, testset_dataset])
                    
        return trainset_dataset, testset_dataset

    if args.dataset == "frei":

        trainset_dataset = make_hand_data_loader(
            args, args.train_yaml, False, is_train=True, scale_factor=args.img_scale_factor)  # RGB image
        testset_dataset = make_hand_data_loader(
            args, args.val_yaml, False, is_train=False, scale_factor=args.img_scale_factor)
        
                    
        return trainset_dataset, testset_dataset

    if args.dataset == "rhd":

        dataset = Rhd(args)
        trainset_dataset, testset_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        
        for iter, degree in enumerate(folder_num):
                ratio  = ((len(trainset_dataset) + len(testset_dataset)) * args.ratio_of_other) / 373184
                dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

                if iter == 0:
                    train_dataset, test_dataset = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

                else:
                    train_dataset_other, test_dataset_other = random_split(
                        dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])
                    test_dataset = ConcatDataset(
                        [test_dataset, test_dataset_other])
                    
        trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
        testset_dataset = ConcatDataset([test_dataset, testset_dataset])
                    
        return trainset_dataset, testset_dataset

    else:
        if not args.general:
            test_dataset = Our_testset_new(args)
            for iter, degree in enumerate(folder_num):

                if iter == 0 :
                    train_dataset = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= args.ratio_of_our)
                
                else:
                    train_dataset_other = CustomDataset(args, degree, path, color=args.color,
                                        ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= args.ratio_of_our)
                    train_dataset = ConcatDataset(
                        [train_dataset, train_dataset_other])

        else:
            folder_num = os.listdir(general_path)
            for iter, degree in enumerate(folder_num):

                if iter == 0:
                    train_dataset = CustomDataset(args, degree, general_path, color=args.color, ratio_of_aug=args.ratio_of_aug, ratio_of_dataset = 1)
                else:
                    dataset = CustomDataset(args, degree, general_path, color=args.color, ratio_of_aug=args.ratio_of_aug, ratio_of_dataset = 1)
                    train_dataset = ConcatDataset([train_dataset, dataset])
            # test_dataset = Frei(args)
            testset_dataset = make_hand_data_loader(
                    args, args.val_yaml, False, is_train=False, scale_factor=args.img_scale_factor)
            # test_dataset = ConcatDataset([test_dataset_general, test_dataset])

    return train_dataset, testset_dataset


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


def apply(img, aug, num=1, scale=1.5):
    Y = [aug(img) for _ in range(num)]
    return Y


def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
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

        meta_list = self.meta['images'].copy()
        index = []
        for idx, j in enumerate(meta_list):
            if j['camera'] == '0':
                index.append(idx)
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            # only one scalar (later u need x,y focal_length)
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(f'{path}/{self.degree}/images/train/{name}')
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

        print(
            f"Done ===> {path}/{degree}/annotations/train/CISLAB_train_data_update.json")
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


class CustomDataset(Dataset):
    def __init__(self, args, degree, path, rotation=False, color=False, ratio_of_aug=0.2, ratio_of_dataset=1):
        self.args = args
        self.rotation = rotation
        self.color = color
        self.degree = degree
        self.path = path
        self.ratio_of_aug = ratio_of_aug
        self.ratio_of_dataset = ratio_of_dataset
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data_update.json", "r") as st_json:
            self.meta = json.load(st_json)

    def __len__(self):
        return int(len(self.meta['images']) * self.ratio_of_dataset)

    def __getitem__(self, idx):

        name = self.meta['images'][idx]['file_name']
        move = self.meta['images'][idx]['move']
        degrees = self.meta['images'][idx]['degree']
        image = cv2.imread(
            f'{self.path}/{self.degree}/images/train/{name}')  # PIL image
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

            # if self.rotation:
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
            # logger.info("Save checkpoint to epoch:{}_{}".format(
            #     epoch, checkpoint_dir))
            break
        except:
            pass
    # else:
    #     logger.info(
    #         "Failed to save checkpoint after {} trails.".format(num_trial))
    return model_to_save, checkpoint_dir


class HIU_Dataset(Dataset):
    def __init__(self, args):
        image_list = []
        for (root, _, files) in os.walk("../../datasets/HIU_DMTL"):
            for file in files:
                if not file.endswith('.json') and not file.endswith('_mask.png') and not file.endswith('_mask.jpg'):
                    file_path = os.path.join(root, file)
                    anno_name = file_path[:-4] + '.json'
                    if os.path.isfile(os.path.join(root, anno_name)):
                        image_list.append((file_path, anno_name))
        self.image = image_list
        self.args = args

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        image = Image.open(self.image[idx][0])
        scale_x = size / image.width
        scale_y = size / image.height

        with open(self.image[idx][1], "r") as st_json:
            annotation = json.load(st_json)

        if annotation['hand_type'][0] == 0:
            joint = annotation['pts2d_2hand'][21:]
        else:
            joint = annotation['pts2d_2hand'][:21]
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        joint_2d = torch.tensor(joint)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d / 2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)

        return trans_image, joint_2d, heatmap, torch.ones(21, 3)


class Our_testset(Dataset):
    def __init__(self, path, folder_name, model):

        self.image_path = f'{path}/{folder_name}/rgb'
        self.anno_path = f'{path}/{folder_name}/annotations'
        self.list = os.listdir(self.image_path)
        self.model = model

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        if not self.model == "ours":
            size = 256
        else:
            size = 224
            
        image = Image.open(os.path.join(self.image_path, self.list[idx]))
        scale_x = size / image.width
        scale_y = size / image.height

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

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        joint_2d = torch.tensor(joint_2d)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        return trans_image, joint_2d

class Our_testset_new(Dataset):
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

        return trans_image, joint_2d_v, heatmap, pose_type


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


class Coco(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/coco/train2017"
        self.datalist = {'img': [], 'kpt': [], 'bbox': []}
        db = COCO("../../datasets/coco/annotations/coco_wholebody_train_v1.0.json")
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann["righthand_valid"]:
                bbox = ann['righthand_box']
                kpt = ann["righthand_kpts"]
                aid = str(ann['image_id'])
                aid = aid.zfill(12)
                img = Image.open(os.path.join(self.img_path, f'{aid}.jpg'))
                if np.array(img).ndim == 3:
                    self.datalist['img'].append(img)
                    self.datalist['kpt'].append(kpt)
                    self.datalist['bbox'].append(bbox)

    def __len__(self):
        return len(self.datalist['img'])

    def __getitem__(self, idx):
        ori_img = self.datalist['img'][idx]

        img = np.array(ori_img)
        joint = np.array(self.datalist['kpt'][idx]).reshape(21, 3)[:, :-1]
        bbox = list(map(int, self.datalist['bbox'][idx]))

        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[0] < 0:
            bbox[0] = 0
            
        img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        joint[:, 0] = joint[:, 0] - bbox[0]
        joint[:, 1] = joint[:, 1] - bbox[1]

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        joint[:, 0] = joint[:, 0] * (size / bbox[2])
        joint[:, 1] = joint[:, 1] * (size / bbox[3])
        joint = torch.tensor(joint)

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint/4)

        img = Image.fromarray(img)
        img = trans(img)

        return img, joint.float(), heatmap, torch.ones(21, 3)


class Panoptic(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/panoptic/hand143_panopticdb"
        with open("../../datasets/panoptic/hand143_panopticdb/hands_v143_14817.json", "r") as st_json:
            self.anno = json.load(st_json)

    def __len__(self):
        return len(self.anno['root'])

    def __getitem__(self, idx):
        ori_img = Image.open(os.path.join(
            self.img_path, self.anno['root'][idx]['img_paths']))
        joint = np.array(self.anno['root'][idx]['joint_self'])
        img = np.array(ori_img)
        hand_center_point = [int(min(joint[:, 0])) + round((max(joint[:, 0]) - min(joint[:, 0]))/2),
                             int(min(joint[:, 1])) + round((max(joint[:, 1]) - min(joint[:, 1]))/2)]

        """ 
            ori_img: 1080 x 1920 size
            so crop the area of hand like 224 x 224
            below code is that it prevents onut of boundary by cropping hand 
        """
        if hand_center_point[0] + 112 > ori_img.width:
            diff_l, hand_center_point[0] = ori_img.width - \
                hand_center_point[0], ori_img.width - 112
        elif hand_center_point[0] - 112 < 0:
            diff_l, hand_center_point[0] = hand_center_point[0], 112
        else:
            diff_l = 112

        if hand_center_point[1] + 112 > ori_img.height:
            diff_d, hand_center_point[1] = ori_img.height - \
                hand_center_point[1], ori_img.height - 112
        elif hand_center_point[1] - 112 < 0:
            diff_d, hand_center_point[1] = hand_center_point[1], 112
        else:
            diff_d = 112

        img = img[hand_center_point[1] - 112: hand_center_point[1] +
                  112, hand_center_point[0] - 112: hand_center_point[0] + 112]

        joint[:, 0] = joint[:, 0] - (hand_center_point[0] - diff_l)
        joint[:, 1] = joint[:, 1] - (hand_center_point[1] - diff_d)

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        joint[:, 0] = joint[:, 0] * (size / 224)
        joint[:, 1] = joint[:, 1] * (size / 224)
        joint = torch.tensor(joint)

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint/4)
        img = Image.fromarray(img)
        img = trans(img)

        return img, joint[:, :2].float(), heatmap, torch.ones(21, 3)


class Rhd(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/rhd"
        with open("../../datasets/rhd/annotations/rhd_train.json", "r") as st_json:
            self.anno = json.load(st_json)
        print()

    def __len__(self):
        return len(self.anno['annotations'])

    def __getitem__(self, idx):
        ori_img = Image.open(os.path.join(
            self.img_path, self.anno['images'][idx]['file_name']))
        joint = np.array(self.anno['annotations'][idx]['keypoints'])
        bbox = list(map(int, self.anno['annotations'][idx]['bbox']))
        img = np.array(ori_img)

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
            
        if bbox[2] % 2 == 1: bbox[2] - 1
        if bbox[3] % 2 == 1: bbox[3] - 1
        space_l = int(224 - bbox[3]) / 2; space_r = int(224 - bbox[2]) / 2
        if (bbox[1] - space_l) < 0: space_l = bbox[1]
        if (bbox[1] + bbox[3] + space_l) > ori_img.height: space_l = ori_img.height - (bbox[1] + bbox[3]) - 1
        if (bbox[0] - space_r) < 0: space_r = bbox[0]
        if (bbox[0] +  bbox[2] + space_r) > ori_img.width: space_r = ori_img.width - (bbox[0] + bbox[2]) - 1
        
        img = img[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        joint[:, 0] = (joint[:, 0] - bbox[0] + space_r) * (self.anno['images'][idx]['width']/(bbox[2] + 2*space_r))
        joint[:, 1] = (joint[:, 1] - bbox[1] + space_l) * (self.anno['images'][idx]['width']/(bbox[3] + 2*space_l))
        
        joint_order = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12,
                       11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        joint = joint[joint_order, :]
        joint[:, 0] = joint[:, 0] * (size / self.anno['images'][idx]['width'])
        joint[:, 1] = joint[:, 1] * (size / self.anno['images'][idx]['height'])
        joint = torch.tensor(joint)

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint/4)
        img = Image.fromarray(img)
        img = trans(img)

        return img, joint[:, :2].float(), heatmap, torch.ones(21, 3)


class Dataset_interhand(torch.utils.data.Dataset):
    def __init__(self, transform, mode, args):
        self.args = args
        self.mode = mode  # train, test, val
        self.img_path = '../../datasets/InterHand2.6M/images'
        self.annot_path = '../../datasets/InterHand2.6M/annotations'
        if self.mode == 'val':
            self.rootnet_output_path = '../../datasets/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = '../../datasets/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(
            0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num*2)}
        self.skeleton = load_skeleton(
            op.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        # load annotation
        print("Load annotation from  " + op.join(self.annot_path, self.mode))
        db = COCO(op.join(self.annot_path, self.mode,
                  'InterHand2.6M_' + self.mode + '_data.json'))
        with open(op.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(op.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
        if (self.mode == 'val' or self.mode == 'test'):
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = op.join(self.img_path, self.mode, img['file_name'])

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(
                frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(
                1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']
                        ] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']
                        ] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array(
                (ann['hand_type_valid']), dtype=np.float32)

            # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            if (self.mode == 'val' or self.mode == 'test'):
                bbox = np.array(
                    rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(
                    aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'],
                                                2], 'left': joint_cam[self.root_joint_idx['left'], 2]}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam,
                     'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint, 'hand_type': hand_type,
                    'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            # if hand_type == 'right' or hand_type == 'left':
            if hand_type == 'right':
                if np.array(Image.open(img_path)).ndim != 3:
                    continue
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        # self.datalist = self.datalist_sh + self.datalist_ih
        self.datalist = self.datalist_sh
        print('Number of annotations in single hand sequences: ' +
              str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' +
              str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data[
            'bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        # 3rd dimension means depth-relative value
        joint = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)

        if self.args.model == "ours":
            size = 224
        else:
            size = 256
        

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])

        ori_img = Image.open(img_path)
        
        bbox = list(map(int, bbox))
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[0] < 0:
            bbox[0] = 0        
        if bbox[2] % 2 == 1: bbox[2] - 1
        if bbox[3] % 2 == 1: bbox[3] - 1
        space_l = int(224 - bbox[3]) / 2; space_r = int(224 - bbox[2]) / 2
        if (bbox[1] - space_l) < 0: space_l = bbox[1]
        if (bbox[1] + bbox[3] + space_l) > ori_img.height: space_l = ori_img.height - (bbox[1] + bbox[3]) - 1
        if (bbox[0] - space_r) < 0: space_r = bbox[0]
        if (bbox[0] +  bbox[2] + space_r) > ori_img.width: space_r = ori_img.width - (bbox[0] + bbox[2]) - 1
        
        # img = img[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        joint[:, 0] = (joint[:, 0] - bbox[0] + space_r) * (ori_img.width/(bbox[2] + 2*space_r))
        joint[:, 1] = (joint[:, 1] - bbox[1] + space_l) * (ori_img.height/(bbox[3] + 2*space_l))
        
        img = np.array(ori_img)[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        img = Image.fromarray(img); img = trans(img)
    
        # reorganize interhand order to ours
        joint = joint[(
            20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16), :]
        # joint[:, 0] = joint[:, 0] - bbox[0]; joint[:, 1] = joint[:, 1] - bbox[1]
        
        joint[:, 0] = joint[:, 0] * (size / ori_img.width)
        joint[:, 1] = joint[:, 1] * (size / ori_img.height)
        targets = torch.tensor(joint[:21, :-1])

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(targets/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(targets/4)

        return img, targets, heatmap, torch.ones(21, 3)
    
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

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d/4)

        return trans_image, joint_2d, heatmap, anno_xyz
    
# def main():
#     path = "../../../../../../data1/1231"
#     path_dir = os.listdir(path)
#     for dir_name in path_dir:
#         if len(dir_name) < 5:
#             Json_transform(path = path, degree = dir_name)
    
# if __name__ =="__main__":
#     main()
#     print("end")