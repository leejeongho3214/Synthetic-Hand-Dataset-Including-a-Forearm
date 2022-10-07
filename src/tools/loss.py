import numpy as np
import torch
from visualize import visualize_with_bbox, visualize_media

def MPJPE(pred_2d_joints, gt_2d_joint):
    batch_size = pred_2d_joints.size(0)
    count = 0
    distance = 0
    for j in range(batch_size):
        for i in range(pred_2d_joints.size(1)):
            align_pred_x = int(pred_2d_joints[j][i][0]) - int(pred_2d_joints[j][0][0])
            align_pred_y = int(pred_2d_joints[j][i][1]) - int(pred_2d_joints[j][0][1])
            align_gt_x = gt_2d_joint[j][i][0] - gt_2d_joint[j][0][0]
            align_gt_y = gt_2d_joint[j][i][1] - gt_2d_joint[j][0][1]
            pred = np.array((align_pred_x, align_pred_y))
            gt = np.array((align_gt_x.detach().cpu(), align_gt_y.detach().cpu()))
            pixel = np.sqrt(np.sum((pred - gt)**2))
            distance += np.sqrt(pixel)
    mpjpe = distance/(batch_size*pred_2d_joints.size(1))

    return mpjpe

def calcu_one(pred_2d_joints, gt_2d_joint):
    distance = 0
    for i in range(21):
        x = int(pred_2d_joints[i][0]) - int(pred_2d_joints[0][0])
        y = int(pred_2d_joints[i][1]) - int(pred_2d_joints[0][1])
        x1 = gt_2d_joint[i][0] - gt_2d_joint[0][0]
        y1 = gt_2d_joint[i][1] - gt_2d_joint[0][1]
        pred_joint_coord = np.sqrt((x ** 2 + y ** 2))
        gt_joint_coord = np.sqrt((x1 ** 2 + y1 ** 2).detach().cpu())
        pixel = (pred_joint_coord - gt_joint_coord) ** 2
        distance += np.sqrt(pixel)
    pred = distance/21

    return pred

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = 1
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
    return loss

def PCK_2d_loss(pred_2d, gt_2d, images, T = 0.1):
    bbox_size = []
    point = []
    pred_2d = pred_2d.detach().cpu()
    gt_2d = gt_2d.detach().cpu()

    for j in gt_2d:
        width = max(j[:,0]) - min(j[:,0])
        height = max(j[:,1]) - min(j[:,1])
        bbox_size.append(max(width, height))
        point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))

    # If you want to show joint with bbox, it can do
    # for k, l, point in zip(images, gt_2d, point):
    #     visualize_with_bbox(k, l, point[0], point[1])
    correct = 0
    visible_joint = 0
    for box_num, box_size in enumerate(bbox_size):
        for joint_num in range(21):
            if gt_2d[box_num][joint_num][2] == 1:
                visible_joint += 1
                x = gt_2d[box_num][joint_num][0] - pred_2d[box_num][joint_num][0]
                y = gt_2d[box_num][joint_num][1] - pred_2d[box_num][joint_num][1]
                distance = np.sqrt((x ** 2 + y ** 2))/box_size
                if distance < T:
                    correct += 1

    return correct, visible_joint

def PCK_2d_loss_HIU(pred_2d, gt_2d, images):
    bbox_size = []
    point = []
    pred_2d = pred_2d.detach().cpu()
    gt_2d = gt_2d.detach().cpu()

    for j in gt_2d:
        width = max(j[:,0]) - min(j[:,0])
        height = max(j[:,1]) - min(j[:,1])
        bbox_size.append(max(width, height))
        point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))

    # If you want to show joint with bbox, it can do
    # for k, l, point in zip(images, gt_2d, point):
    #     visualize_with_bbox(k, l, point[0], point[1])
    correct = 0
    visible_joint = 0
    T = 0.05
    for box_num, box_size in enumerate(bbox_size):
        for joint_num in range(21):
            visible_joint += 1
            x = gt_2d[box_num][joint_num][0] - pred_2d[box_num][joint_num][0]
            y = gt_2d[box_num][joint_num][1] - pred_2d[box_num][joint_num][1]
            distance = np.sqrt((x ** 2 + y ** 2))/box_size
            if distance < T:
                correct += 1

    return correct, visible_joint

def PCK_2d_loss_No_batch(pred_2d, gt_2d, images,T=0.1, file = None):
    point = []
    pred_2d = pred_2d.detach().cpu()
    gt_2d = gt_2d.detach().cpu()

    width = max(gt_2d[:, 0]) - min(gt_2d[:, 0])
    height = max(gt_2d[:, 1]) - min(gt_2d[:, 1])
    box_size = max(width, height)
    point.append(((min(gt_2d[:, 0]), min(gt_2d[:, 1])), (max(gt_2d[:, 0]), max(gt_2d[:, 1]))))

    # If you want to show joint with bbox, it can do
    # for k, l, point in zip(images, gt_2d, point):
    #     visualize_with_bbox(k, l, point[0], point[1])
    correct = 0
    visible_joint = 0

    for joint_num in range(21):
        if gt_2d[joint_num][2] == 1:
            visible_joint += 1
            x = gt_2d[joint_num][0] - pred_2d[joint_num][0]
            y = gt_2d[joint_num][1] - pred_2d[joint_num][1]
            distance = np.sqrt((x ** 2 + y ** 2))/box_size
            if distance < T:
                correct += 1
    if (correct/visible_joint) < 0.6:
        # visualize_with_bbox(images, pred_2d, point[0][0], point[0][1], file)
        from matplotlib import pyplot as plt
        fig = plt.figure()
        visualize_media(images, pred_2d, gt_2d,file, fig)
    return correct, visible_joint


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs / 2.0)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    # gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    # conf = conf[has_pose_3d == 1]
    # pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        # return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()