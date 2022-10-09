
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

def visualize_prediction_show(images, pred_2d_joint, fig):
    image = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])),
                     (int(pred_2d_joint[0][parents[i]][0]), int(pred_2d_joint[0][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('pred_image')
    ax1.axis("off")
    plt.show()

def visualize_prediction(images, pred_2d_joint, fig, epoch,iteration,args):

    image = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])),
                     (int(pred_2d_joint[0][parents[i]][0]), int(pred_2d_joint[0][parents[i]][1])),
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
        plt.savefig(f"eval_image/{args.name[7:-31]}/{iteration}.jpg")

    else:
        if os.path.isdir("test_image") == False:
            mkdir("test_image")
        if os.path.isdir(f'test_image/{args.name}') == False:
            mkdir(f'test_image/{args.name}')
        if os.path.isdir(f'test_image/{args.name}/{epoch}_epoch') == False:
            mkdir(f'test_image/{args.name}/{epoch}_epoch')
        plt.savefig(f"test_image/{args.name}/{epoch}_epoch/{iteration}_iter.jpg")


def visualize_gt(images, gt_2d_joint, fig):
    # for j in gt_2d:
    #     width = max(j[:,0]) - min(j[:,0])
    #     height = max(j[:,1]) - min(j[:,1])
    #     bbox_size.append(max(width, height))
    #     point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))

    image = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[0][i][0]), int(gt_2d_joint[0][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[0][i][0]), int(gt_2d_joint[0][i][1])),
                     (int(gt_2d_joint[0][parents[i]][0]), int(gt_2d_joint[0][parents[i]][1])),
                     [0, 0, 1], 1)
        # if i == 20:
        #     cv2.rectangle(image, (int(box_left_bottom[0]),int(box_left_bottom[1])), (int(box_right_top[0]),int(box_right_top[1])), (0, 255, 0), 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('gt_image')
    ax1.axis("off")

def visualize_with_bbox(images, gt_2d_joint, box_left_bottom, box_right_top, file):
    # image = np.moveaxis(images, 0, -1)
    image = images
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])),
                     (int(gt_2d_joint[parents[i]][0]), int(gt_2d_joint[parents[i]][1])),
                     [0, 0, 1], 1)
        if i == 20:
            cv2.rectangle(image, (int(box_left_bottom[0]),int(box_left_bottom[1])), (int(box_right_top[0]),int(box_right_top[1])), (0, 255, 0), 1)

    plt.imshow(image[:,:,[2,1,0]])
    import os

    plt.savefig(f'mediapipe_error/{file}', dpi = 300)
    # plt.show()

def visualize_our(image_path, json_path, trans):
    image = Image.open(image_path)

    image = trans(image)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        joint_total = json_data['annotations']
        joint = {}
        joint_2d = []
        for j in joint_total:
            if j['label'] != 'Pose':
                if len(j['metadata']['system']['attributes']) > 0:
                    # Change 'z' to 'indicator function'
                    # Ex. 0 means visible joint, 1 means invisible joint
                    j['coordinates']['z'] = 1
                    joint[f"{int(j['label'])}"] = j['coordinates']
                else:
                    joint[f"{int(j['label'])}"] = j['coordinates']

        for h in range(0, 21):
            joint_2d.append([joint[f'{h}']['x'], joint[f'{h}']['y'], joint[f'{h}']['z']])

    joint_2d = torch.tensor(joint_2d)
    # visualize(image, joint_2d)

def visualize(image, gt_2d_joint):
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])),
                     (int(gt_2d_joint[parents[i]][0]), int(gt_2d_joint[parents[i]][1])),
                     [0, 0, 1], 1)

    cv2.imwrite('vis.jpg', image)

def visualize_HIU(image_path, json_path, trans):

    image = Image.open(image_path)
    alpha = 224/ image.height

    image = trans(image)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        joint_total = json_data['pts2d_2hand']

        joint_2d = []
        for j in joint_total:
            if j[0] != 2.0:
                joint_2d.append(j)

    joint_2d = torch.tensor(joint_2d) * alpha
    visualize(image, joint_2d)

def visualize_media(image, pred_2d_joint, gt_2d_joint, file_name, fig):
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    image1 = image.copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, ])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[i][0]), int(pred_2d_joint[i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[i][0]), int(pred_2d_joint[i][1])),
                     (int(pred_2d_joint[parents[i]][0]), int(pred_2d_joint[parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('pred_image')
    ax1.axis("off")
    # plt.show()

    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, ])
    for i in range(21):
        cv2.circle(image1, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image1, (int(gt_2d_joint[i][0]), int(gt_2d_joint[i][1])),
                     (int(gt_2d_joint[parents[i]][0]), int(gt_2d_joint[parents[i]][1])),
                     [0, 0, 1], 1)
        # if i == 20:
        #     cv2.rectangle(image, (int(box_left_bottom[0]),int(box_left_bottom[1])), (int(box_right_top[0]),int(box_right_top[1])), (0, 255, 0), 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1[:, :, (2, 1, 0)])
    ax1.set_title('gt_image')
    ax1.axis("off")
    plt.savefig(f'mediapipe_error/{file_name}', dpi=300)

def main():
    for i in range(0,100):
        visualize_our(f'../../datasets/our_testset/rgb/0.jpg', f'../../datasets/our_testset/annotation/{i}.json', trans)



if __name__ == "__main__":
    main()
