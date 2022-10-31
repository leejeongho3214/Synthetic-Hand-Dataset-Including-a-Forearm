
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import json
from src.utils.miscellaneous import mkdir
from PIL import Image
from torchvision.transforms import transforms

trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def visualize_prediction_media(images, pred_2d_joint, fig, epoch,iteration,args =None, dataset_name = 'p'):
    # num = iteration % images.size(0)
    num =0 
    # image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = images
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])),
                     (int(pred_2d_joint[num][parents[i]][0]), int(pred_2d_joint[num][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('pred_image')
    ax1.axis("off")
    if epoch == 'evaluation':
        if os.path.isdir("eval_image") == False:
            mkdir("eval_image")
        if os.path.isdir(f'eval_image/{args.name[7:-31]}') == False:
            mkdir(f'eval_image/{args.name[7:-31]}')
        if os.path.isdir(f'eval_image/{args.name[7:-31]}/{dataset_name}') == False:
            mkdir(f'eval_image/{args.name[7:-31]}/{dataset_name}')
        plt.savefig(f"eval_image/{args.name[7:-31]}/{dataset_name}/{iteration}.jpg")

    elif epoch == 'mediapipe':
        if os.path.isdir("eval_image") == False:
            mkdir("eval_image")
        if os.path.isdir('eval_image/mediapipe') == False:
            mkdir('eval_image/mediapipe')
        plt.savefig(f"eval_image/mediapipe/{iteration}.jpg")

    else:
        if os.path.isdir("test_image") == False:
            mkdir("test_image")
        if os.path.isdir(f'test_image') == False:
            mkdir(f'test_image')
        if os.path.isdir(f'test_image/{epoch}_epoch') == False:
            mkdir(f'test_image/{epoch}_epoch')
        plt.savefig(f"test_image/{epoch}_epoch/{iteration}_iter.jpg")


def visualize_gt_media(images, gt_2d_joint, fig, num):
    # for j in gt_2d:
    #     width = max(j[:,0]) - min(j[:,0])
    #     height = max(j[:,1]) - min(j[:,1])
    #     bbox_size.append(max(width, height))
    #     point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))
    # num = num % images.size(0)
    num = 0
    # image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = images
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])),
                     (int(gt_2d_joint[num][parents[i]][0]), int(gt_2d_joint[num][parents[i]][1])),
                     [0, 0, 1], 1)
        # if i == 20:
        #     cv2.rectangle(image, (int(box_left_bottom[0]),int(box_left_bottom[1])), (int(box_right_top[0]),int(box_right_top[1])), (0, 255, 0), 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('gt_image')
    ax1.axis("off")

def visualize_prediction(images, pred_2d_joint, fig, method = None, epoch = 0, iteration = 0,args =None, dataset_name = 'p'):
    num = iteration % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])),
                     (int(pred_2d_joint[num][parents[i]][0]), int(pred_2d_joint[num][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image)
    ax1.set_title('pred_image')
    ax1.axis("off")

    if method == 'evaluation':
        os.remove(f"{args.output_dir}/eval_image")
        if os.path.isdir(f"eval_image") == False:
            mkdir("eval_image")
        if os.path.isdir(f'eval_image/{args.name[7:-31]}') == False:
            mkdir(f'eval_image/{args.name[7:-31]}')
        if os.path.isdir(f'eval_image/{args.name[7:-31]}/{dataset_name}') == False:
            mkdir(f'eval_image/{args.name[7:-31]}/{dataset_name}')
        plt.savefig(f"eval_image/{args.name[7:-31]}/{dataset_name}/{iteration}.jpg")

    elif method == 'mediapipe':
        os.remove(f"{args.output_dir}/eval_image")
        if os.path.isdir("eval_image") == False:
            mkdir("eval_image")
        if os.path.isdir('eval_image/mediapipe') == False:
            mkdir('eval_image/mediapipe')
        plt.savefig(f"eval_image/mediapipe/{iteration}.jpg")

    elif method == 'train':
        os.remove(f"{args.output_dir}/train_image")
        if os.path.isdir(f"{args.output_dir}/train_image") == False:
            mkdir(f"{args.output_dir}/train_image")
        if os.path.isdir(f'{args.output_dir}/train_image/{epoch}_epoch') == False:
            mkdir(f'{args.output_dir}/train_image/{epoch}_epoch')
        plt.savefig(f"{args.output_dir}/train_image/{epoch}_epoch/{iteration}_iter.jpg")
    else:
        os.remove(f"{args.output_dir}/test_image")
        if os.path.isdir(f"{args.output_dir}/test_image") == False:
            mkdir(f"{args.output_dir}/test_image")
        if os.path.isdir(f'{args.output_dir}/test_image/{epoch}_epoch') == False:
            mkdir(f'{args.output_dir}/test_image/{epoch}_epoch')
        plt.savefig(f"{args.output_dir}/test_image/{epoch}_epoch/{iteration}_iter.jpg")


def visualize_gt(images, gt_2d_joint, fig, num):
    # for j in gt_2d:
    #     width = max(j[:,0]) - min(j[:,0])
    #     height = max(j[:,1]) - min(j[:,1])
    #     bbox_size.append(max(width, height))
    #     point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))
    num = num % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])),
                     (int(gt_2d_joint[num][parents[i]][0]), int(gt_2d_joint[num][parents[i]][1])),
                     [0, 0, 1], 1)
        # if i == 20:
        #     cv2.rectangle(image, (int(box_left_bottom[0]),int(box_left_bottom[1])), (int(box_right_top[0]),int(box_right_top[1])), (0, 255, 0), 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('gt_image')
    ax1.axis("off")
