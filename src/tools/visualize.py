
import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.miscellaneous import mkdir
from torchvision.transforms import transforms

def visualize_pred(images, pred_2d_joint, fig, method = None, epoch = 0, iteration = 0, args =None, dataset_name = 'p'):

    num = iteration % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
<<<<<<< HEAD
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
=======
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
>>>>>>> 75095027a1e1ec114691fcfe220c154ef0b276bb
    
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
        if iteration == 0 and epoch == 0:
            if os.path.isdir(f'eval_image/{args.name[13:-31]}/{dataset_name}') == True:
                shutil.rmtree(f"eval_image/{args.name[13:-31]}/{dataset_name}")
        if os.path.isdir(f'eval_image/{args.name[13:-31]}/{dataset_name}') == False:
            mkdir(f'eval_image/{args.name[13:-31]}/{dataset_name}')
        plt.savefig(f"eval_image/{args.name[13:-31]}/{dataset_name}/{iteration}.jpg")

    elif method == 'mediapipe':
        if iteration == 0 and epoch == 0:
            if os.path.isdir(f"eval_image/mediapipe"):
                shutil.rmtree(f"eval_image")
        if os.path.isdir('eval_image/mediapipe') == False:
            mkdir('eval_image/mediapipe')
        plt.savefig(f"eval_image/mediapipe/{iteration}.jpg")

    elif method == 'train':
        if iteration == 0 and epoch == 0:
            if os.path.isdir(f"{args.output_dir}/train_image"):
                shutil.rmtree(f"{args.output_dir}/train_image")
        if os.path.isdir(f'{args.output_dir}/train_image/{epoch}_epoch') == False:
            mkdir(f'{args.output_dir}/train_image/{epoch}_epoch')
        plt.savefig(f"{args.output_dir}/train_image/{epoch}_epoch/{iteration}_iter.jpg")

    elif method == 'test':
        if iteration == 0 and epoch == 0:
            if os.path.isdir(f"{args.output_dir}/test_image"):
                shutil.rmtree(f"{args.output_dir}/test_image")
        if os.path.isdir(f'{args.output_dir}/test_image/{epoch}_epoch') == False:
            mkdir(f'{args.output_dir}/test_image/{epoch}_epoch')
        plt.savefig(f"{args.output_dir}/test_image/{epoch}_epoch/{iteration}_iter.jpg")

    else:
        assert False, "method is the wrong name"

def visualize_gt(images, gt_2d_joint, fig, iteration):

    num = iteration % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
<<<<<<< HEAD
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
=======
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
>>>>>>> 75095027a1e1ec114691fcfe220c154ef0b276bb
    
    for i in range(21):

        cv2.circle(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])), 2, [0, 1, 0],
                    thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])),
                     (int(gt_2d_joint[num][parents[i]][0]), int(gt_2d_joint[num][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('gt_image')
    ax1.axis("off")

