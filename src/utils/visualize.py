
import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.miscellaneous import mkdir
from src.utils.dir import reset_folder

def visualize_pred(images, pred_2d_joint, fig, method = None, epoch = 0, iteration = 0, args =None):
    for num in range(16):
        image = visualize(images, pred_2d_joint, num)
        ax1 = fig.add_subplot(6, 6, (num+1) * 2)
        
        ax1.imshow(image)
        ax1.axis("off")
    
    if method == 'evaluation':
        if not os.path.isdir("eval_image"): mkdir("eval_image")
        plt.savefig(os.path.join("eval_image", f'iter_{iteration}.jpg') , dpi = 2000)

    elif method == 'train':
        root = f"{args.output_dir}/train_image"
        epoch_path = os.path.join(root, f"{epoch}_epoch")
        if iteration == 0 and epoch == 0:
            reset_folder(root)
            reset_folder("/".join(root.split('/')[:-1]) + "/test_image")
        if not os.path.isdir(epoch_path): mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f'iter_{iteration}.jpg') , dpi = 2000)

    elif method == 'test':
        epoch_path = f'{args.output_dir}/test_image/{epoch}_epoch'
        if not os.path.isdir(epoch_path): mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f'iter_{iteration}.jpg') , dpi = 2000)

    else:
        assert False, "method is the wrong name"

def visualize_gt(images, gt_2d_joint, fig):
    for num in range(16):
        image = visualize(images, gt_2d_joint, num)
        ax1 = fig.add_subplot(6, 6, (num+1) * 2  - 1)
        
        ax1.imshow(image)
        ax1.axis("off")
    
def visualize(images, joint_2d, num):
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    for i in range(21):
        cv2.circle(image, (int(joint_2d[num][i][0]), int(joint_2d[num][i][1])), 2, [0, 1, 0],
                    thickness=-1)
        if i != 0:
            cv2.line(image, (int(joint_2d[num][i][0]), int(joint_2d[num][i][1])),
                        (int(joint_2d[num][parents[i]][0]), int(joint_2d[num][parents[i]][1])),
                        [0, 0, 1], 1)
            
    return image


def visualize_only_gt(images, gt_2d_joint, fig, method = None, epoch = 0, iteration = 0, args =None):
    
    for num in range(min(len(images), 16)):
        image = visualize(images, gt_2d_joint, num)
        ax1 = fig.add_subplot(4, 4, num + 1)
        
        ax1.imshow(image)
        ax1.axis("off")
    
    if method == "train":
        root = f"{args.output_dir}/train_image"
    else:
        root = f"{args.output_dir}/val_image"
        
    epoch_path = os.path.join(root, f"{epoch}_epoch")
    if iteration == 0 and epoch == 0:
        reset_folder(root)
    if not os.path.isdir(epoch_path): mkdir(epoch_path)
    plt.savefig(os.path.join(epoch_path, f'iter_{iteration}.jpg') , dpi = 2000)


