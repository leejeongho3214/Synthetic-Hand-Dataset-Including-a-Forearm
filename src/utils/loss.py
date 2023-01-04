from cv2 import threshold
import numpy as np
import torch
from src.utils.visualize import *

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
    if gt_keypoints_2d.size(2) > 2:
        loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, : ,:2])) * gt_keypoints_2d[:, : ,2][:, :, None]      ## It consider to calculate only the visible joint  
        return loss[loss>0].mean()
    else:
        loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, : ,:2])).mean(2).mean(1).sum()
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
    pck = (euclidean_dist * pixel_to_mm <= T).type(torch.float).mean()

    return pck, T

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
        
def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize
    
def calc_loss(combined_hm_preds, heatmaps, args):

    if args.model == "hourglass": 
        combined_loss = []
        for i in range(8):
            combined_loss.append(HeatmapLoss()(combined_hm_preds[:, i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
    else: combined_loss = HeatmapLoss()(combined_hm_preds, heatmaps)
    
    return combined_loss


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re
