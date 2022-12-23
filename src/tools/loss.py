from cv2 import threshold
import numpy as np
import torch
from visualize import *

def MPJPE_visible(pred_2d_joints, gt_2d_joint):
    batch_size = pred_2d_joints.size(0)
    distance = 0
    for j in range(batch_size):
        for i in range(pred_2d_joints.size(1)):
            if gt_2d_joint[j][i][2] == 1:
                assert gt_2d_joint[j][0][2] == 1, "wrist joint is not visible"

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
    

def EPE(pred_2d_joints, gt_2d_joint):
    batch_size = pred_2d_joints.size(0)
    distance = {}
    pred_2d_joints, gt_2d_joint = pred_2d_joints.detach().cpu(), gt_2d_joint.detach().cpu()
    for i in range(1, pred_2d_joints.size(1)):  ## skip the wrist joint
        error = []
        for j in range(batch_size):
            if gt_2d_joint[j, i, 2] == 0: ## invisible joint = 0
                continue    
            pred = pred_2d_joints[j, i]
            gt = gt_2d_joint[j, i, :2]
            error.append(torch.sqrt(torch.sum((pred - gt)**2))) 

        distance[f'{i}'] = [np.mean(np.array(error)) if not np.isnan(np.mean(np.array(error))) else 0, len(error)]

    # epe = [distance[f'{i}'][0] for i in range(len(distance))]
    epe = [[distance[f'{i}'][0] * distance[f'{i}'][1], distance[f'{i}'][1]]  for i in range(1, len(distance))]
    # epe_loss = np.sum(np.array(epe)[:,0])/np.sum(np.array(epe)[:,1]) ## mean every joint

    return (np.sum(np.array(epe)[:,0]), np.sum(np.array(epe)[:,1])), distance


def EPE_train(pred_2d_joints, gt_2d_joint):
    batch_size = pred_2d_joints.size(0)
    distance = {}
    pred_2d_joints, gt_2d_joint = pred_2d_joints.detach().cpu(), gt_2d_joint.detach().cpu()
    for i in range(1, pred_2d_joints.size(1)):
        error = []
        for j in range(batch_size):
            pred = pred_2d_joints[j, i]
            gt = gt_2d_joint[j, i, :2]
            error.append(torch.sqrt(torch.sum((pred - gt)**2))) 

        distance[f'{i}'] = [np.mean(np.array(error)) if not np.isnan(np.mean(np.array(error))) else 0, len(error)]

    # epe = [distance[f'{i}'][0] for i in range(len(distance))]
    epe = [[distance[f'{i}'][0] * distance[f'{i}'][1], distance[f'{i}'][1]]  for i in range(1, len(distance))]
    # epe_loss = np.sum(np.array(epe)[:,0])/np.sum(np.array(epe)[:,1]) ## mean every joint

    return (np.sum(np.array(epe)[:,0]), np.sum(np.array(epe)[:,1])), distance

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = 1
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean(2).mean(1).sum()
    return loss

def PCK_2d_loss_visible(pred_2d, gt_2d, T = 0.1, threshold = 'proportion'):
    bbox_size = []
    point = []
    pred_2d = pred_2d.detach().cpu()  
    gt_2d = gt_2d.detach().cpu()

    for j in gt_2d:
        width = max(j[:,0]) - min(j[:,0])
        height = max(j[:,1]) - min(j[:,1])
        length = np.sqrt( width ** 2 + height ** 2)
        bbox_size.append(length)
        point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))

    gt_2d_value = gt_2d[:, 1:, :-1]; gt_2d_vis = gt_2d[:, 1: ,-1]; pred_2d_value = pred_2d[:, 1:] ## Excluded the wrist joint by starting 1
    vis_index = gt_2d_vis == 1
    diff = gt_2d_value - pred_2d_value
    distance = diff.square().sum(2).sqrt() * vis_index                                            ## Consider only visible joint
    num_vis = len(vis_index[vis_index == True])
    
    if threshold == 'proportion':
        norm_distance = distance.permute(1, 0) / torch.tensor(bbox_size)
        num_correct = num_vis - len(norm_distance[norm_distance > T])
            
    elif threshold == 'mm':
        num_correct = num_vis - len(distance[distance > (T * 3.78)])

    else:
        assert False, "Please check variable threshold is right"

    
    pck = num_correct / num_vis
    return pck

def PCK_2d_loss(pred_2d, gt_2d, T = 0.1, threshold = 'proportion'):
    bbox_size = []
    point = []
    pred_2d = pred_2d.detach().cpu()
    gt_2d = gt_2d.detach().cpu()

    for j in gt_2d:
        width = max(j[:,0]) - min(j[:,0])
        height = max(j[:,1]) - min(j[:,1])
        length = np.sqrt( width ** 2 + height ** 2)
        bbox_size.append(length)
        point.append(((min(j[:,0]),min(j[:,1])),(max(j[:,0]),max(j[:,1]))))

    gt_2d_value = gt_2d[:, 1:]; pred_2d_value = pred_2d[:, 1:] ## Excluded the wrist joint by starting 1
    diff = gt_2d_value - pred_2d_value
    distance = diff.square().sum(2).sqrt()
    num_total = len(distance.flatten())
   
    if threshold == 'proportion':
        norm_distance = distance.permute(1, 0) / torch.tensor(bbox_size)
        num_correct = num_total - len(norm_distance[norm_distance > T])
            
    elif threshold == 'mm':
        num_correct = num_total - len(distance[distance > (T * 3.78)])

    else:
        assert False, "Please check variable threshold is right"

    pck = num_correct / num_total
    
    return pck

def PCK_3d_loss(pred_3d, gt_3d, T = 0.1):
    
    pred_3d = pred_3d.detach().cpu()
    gt_3d = gt_3d.detach().cpu()
    diff = pred_3d - gt_3d
    euclidean_dist = diff.square().sum(2).sqrt()
    pixel_to_mm = 3.779527559
    pck = (euclidean_dist * pixel_to_mm <= T).type(torch.float).mean(1).sum()

    return pck, T

def PCK_2d_loss_No_batch(pred_2d, gt_2d, images,T=0.1, threshold = 'proportion'):
    point = []
    pred_2d = pred_2d.detach().cpu()
    gt_2d = gt_2d.detach().cpu()

    width = max(gt_2d[:, 0]) - min(gt_2d[:, 0])
    height = max(gt_2d[:, 1]) - min(gt_2d[:, 1])
    square_len = width ** 2 + height ** 2
    box_size = np.sqrt(square_len)
    point.append(((min(gt_2d[:, 0]), min(gt_2d[:, 1])), (max(gt_2d[:, 0]), max(gt_2d[:, 1]))))

    # If you want to show joint with bbox, it can do
    # for k, l, point in zip(images, gt_2d, point):
    #     visualize_with_bbox(k, l, point[0], point[1])
    correct = 0
    visible_joint = 0

    for joint_num in range(1, 21):
        if gt_2d[joint_num][2] == 1:
            visible_joint += 1
            x = gt_2d[joint_num][0] - pred_2d[joint_num][0]
            y = gt_2d[joint_num][1] - pred_2d[joint_num][1]
            
            if threshold == 'proportion':
                distance = np.sqrt((x ** 2 + y ** 2))
                if distance < T * box_size:
                    correct += 1
            elif threshold == 'pixel':
                distance = np.sqrt((x ** 2 + y ** 2))
                if distance * 0.26 < T * 20:
                    correct += 1
            else:
                assert False, "Please check variable threshold is right"

    return correct, visible_joint , threshold


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
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean(2).mean(1).sum()
    else:
        assert False, "gt_3d_keypoint No"
