import numpy as np
import torch


def calcu(pred_2d_joints, gt_2d_joint):
    batch_size = pred_2d_joints.size(0)
    count = 0
    distance = 0
    for j in range(batch_size):
        for i in range(pred_2d_joints.size(1)):
            x = int(pred_2d_joints[j][i][0]) - int(pred_2d_joints[j][0][0])
            y = int(pred_2d_joints[j][i][1]) - int(pred_2d_joints[j][0][1])
            x1 = gt_2d_joint[j][i][0] - gt_2d_joint[j][0][0]
            y1 = gt_2d_joint[j][i][1] - gt_2d_joint[j][0][1]
            pred_joint_coord = np.sqrt((x ** 2 + y ** 2))
            gt_joint_coord = np.sqrt((x1 ** 2 + y1 ** 2).detach().cpu())
            pixel = (pred_joint_coord - gt_joint_coord) ** 2
            distance += np.sqrt(pixel)
    pred = distance/(batch_size*pred_2d_joints.size(1))

    return pred

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
