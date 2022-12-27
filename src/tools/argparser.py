import sys
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
import argparse
import torchvision.models as models
from src.modeling.hrnet.config.default import update_config
from src.modeling.hrnet.config.default import _C as cfg
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_pose_net as get_cls_net_gridfeat
from src.modeling.our_hrnet import config as hrnet_config
from src.modeling.our_hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat as get_cls_net_gridfeat_our
from src.modeling.our_hrnet import update_config_our as hrnet_update_config
from src.utils.comm import get_rank
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir
from src.modeling.simplebaseline.config import config as config_simple
from src.modeling.simplebaseline.pose_resnet import get_pose_net
import torch
import os
import gc
from src.datasets.build import make_hand_data_loader
import time
import numpy as np
from matplotlib import pyplot as plt
import torch
from loss import *
from src.utils.geometric_layers import *
from src.utils.metric_logger import AverageMeter
from visualize import *
from time import ctime
from src.modeling.hourglass.posenet import PoseNet

def parse_args():
    parser = argparse.ArgumentParser()
    ######################################################################################
    ## Set Hyper-parameter ##
    ######################################################################################
    parser.add_argument("--name", default='None',
                        help = '20k means CISLAB 20,000 images',type=str)
    parser.add_argument("--root_path", default=f'output', type=str, required=False,
                        help="The root directory to save location which you want")
    parser.add_argument("--model", default='ours', type=str, required=False,
                        help="you can choose model like hrnet, simplebaseline, hourglass, ours")
    parser.add_argument("--dataset", default='ours', type=str, required=False,
                        help="you can choose dataset like ours, coco, interhand, rhd")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--ratio_of_our", default=0.3, type=float)
    parser.add_argument("--ratio_of_other", default=0.3, type=float)
    parser.add_argument("--ratio_of_aug", default=0.2, type=float)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--loss_2d", default=1, type=int)
    parser.add_argument("--loss_3d", default=1, type=int)
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--rot", action='store_true')
    parser.add_argument("--color", action='store_true')
    parser.add_argument("--general", action='store_true')
    parser.add_argument("--projection", action='store_true')
    
    ######################################################################################

    ######################################################################################

    parser.add_argument("--multiscale_inference", default=False, action='store_true', )
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument("--aml_eval", default=False, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X steps.")
    parser.add_argument("--resume_path", default='HIU', type=str)
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.5, type=float)
    parser.add_argument("--vloss_w_sub", default=0.5, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='../../samples/unity/images/train/Capture0', type=str,
                        help="test data")
    parser.add_argument("--train_yaml", default='../../datasets/freihand/train.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--val_yaml", default='../../datasets/freihand/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--model_name_or_path", default='../modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                                "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2048,512,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--which_gcn", default='0,0,1', type=str,
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv")
    parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand")
    parser.add_argument("--run_eval_only", default=True, action='store_true', )
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    
    args = parser.parse_args()
    return args

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

def load_model(args):
    epo = 0
    best_loss = np.inf
    count = 0
    
    args.output_dir = os.path.join(args.root_path, args.name)
    if not args.resume: args.resume_checkpoint = 'None'
    else: args.resume_checkpoint = os.path.join(os.path.join(args.root_path, args.name),'checkpoint-good/state_dict.bin')
    if not os.path.isdir(args.output_dir): mkdir(args.output_dir)
    if os.path.isfile(os.path.join(args.output_dir, "log.txt")): os.remove(os.path.join(args.output_dir, "log.txt"))
    
    logger = setup_logger(args.name, args.output_dir, get_rank())
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.device = torch.device(args.device)

    if args.model == "hrnet":       ## output: 21 x 128 x 128
        update_config(cfg, args)
        _model = get_cls_net_gridfeat(cfg, is_train=True)
        
    elif args.model == 'hourglass': ## output: 21 x 64 x 64
        _model = PoseNet(nstack=8, inp_dim=256, oup_dim= 21, num_parts=args.batch_size, increase=0)
        
    elif args.model == 'simplebaseline': ## output: 21 x 64 x 64
        _model = get_pose_net(config_simple, is_train=True)
        
    else:
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3] ## origin => change to input_feat_dim

        # which encoder block to have graph convs
        which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                                                    else args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size * 2)

            if which_blk_graph[i] == 1:
                config.graph_conv = True
                # logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    # logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            trans_encoder.append(model)

        # create backbone model
        if args.arch == 'hrnet':
            hrnet_yaml = '../../models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = '../../models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat_our(hrnet_config, pretrained=hrnet_checkpoint)
            # logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch == 'hrnet-w64':
            hrnet_yaml = '../../models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = '../../models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat_our(hrnet_config, pretrained=hrnet_checkpoint)
            # logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder, token = 70)

    if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
        state_dict = torch.load(args.resume_checkpoint)
        best_loss = state_dict['best_loss']
        epo = state_dict['epoch']
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)
        # logger.info("Resume: Loading from checkpoint {}\n".format(args.resume_checkpoint))
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    _model.to(args.device)
    return _model, logger, best_loss, epo, count



def train(args, train_dataloader, Graphormer_model, epoch, best_loss, data_len ,logger, count, writer, pck, len_total, batch_time):

    optimizer = torch.optim.Adam(params=list(Graphormer_model.parameters()),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)

    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    end = time.time()
    Graphormer_model.train()
    log_losses = AverageMeter()
    log_2d_losses = AverageMeter()
    log_3d_losses = AverageMeter()
    log_3d_re_losses = AverageMeter()
    
    if args.model == "ours":
        for iteration, (images, gt_2d_joints, heatmap, gt_3d_joints) in enumerate(train_dataloader):
            batch_size = images.size(0)
            adjust_learning_rate(optimizer, epoch, args)  
            gt_2d_joints[:,:,1] = gt_2d_joints[:,:,1] / images.size(2) ## You Have to check whether weight and height is correct dimenstion
            gt_2d_joints[:,:,0] = gt_2d_joints[:,:,0] / images.size(3)  ## 2d joint value rearrange from 0 to 1
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()
            gt_3d_joints = gt_3d_joints.clone().detach()
            gt_3d_joints = gt_3d_joints.cuda()
            images = images.cuda()
            
            if args.projection: pred_2d_joints, pred_3d_joints= Graphormer_model(images)
            else: pred_2d_joints= Graphormer_model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); args.loss_3d = 0
            
            loss_2d= keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joint)
            loss_3d = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
            loss_3d_re = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
            loss = args.loss_2d * loss_2d + args.loss_3d * loss_3d + loss_3d_re
            log_losses.update(loss.item(), batch_size)
            log_2d_losses.update(loss_2d.item(), batch_size)
            log_3d_losses.update(loss_3d.item(), batch_size)
            log_3d_re_losses.update(loss_3d_re.item(), batch_size)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)
            gt_2d_joint[:,:,1] = gt_2d_joint[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            gt_2d_joint[:,:,0] = gt_2d_joint[:,:,0] * images.size(3) 
            
            if not args.projection:
                if iteration == 0 or iteration == int(len(train_dataloader)/2) or iteration == len(train_dataloader) - 1:
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig, iteration)
                    visualize_prediction(images, pred_2d_joints, fig, 'train', epoch, iteration, args, None)
                    plt.close()

            batch_time.update(time.time() - end)
            end = time.time()
            eta_seconds = batch_time.avg * ((len_total - iteration) + (args.epoch - epoch -1) * len_total)  

            if iteration == len(train_dataloader) - 1:
                logger.info(
                    ' '.join(
                        ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi},  count: {count}/{max_count}']
                    ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = args.count)
                    + ' 2d_loss: {:.8f}, 3d_loss: {:.8f}, 3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}\n'.format(
                        log_2d_losses.avg,
                        log_3d_losses.avg,
                        log_3d_re_losses.avg,
                        pck,
                        log_losses.avg,
                        best_loss,
                        ctime(eta_seconds + end))
                )
                writer.add_scalar("Loss/train", log_losses.avg, epoch)

            else:
                logger.info(
                    ' '.join(
                        ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi},  count: {count}/{max_count}']
                    ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = args.count)
                     + ' 2d_loss: {:.8f}, 3d_loss: {:.8f} ,3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                        log_2d_losses.avg,
                        log_3d_losses.avg,
                        log_3d_re_losses.avg,
                        pck,
                        log_losses.avg,
                        best_loss,
                        ctime(eta_seconds + end))
                )
                
        return Graphormer_model, optimizer, batch_time
    
    else:
        heatmap_size, multiply = 64, 4
        if args.model == "hrnet": heatmap_size, multiply = 128, 2
        for iteration, (images, gt_2d_joints, gt_heatmaps, gt_3d_joints) in enumerate(train_dataloader):
            batch_time = AverageMeter()
            batch_size = images.size(0)
            adjust_learning_rate(optimizer, epoch, args)
            images = images.cuda()

            gt_heatmaps = gt_heatmaps.cuda()     
            pred = Graphormer_model(images)
   
            loss = torch.mean(calc_loss(pred, gt_heatmaps, args))

            log_losses.update(loss.item(), batch_size)

            if args.model == "hourglass": pred = pred[:, -1]

            pred_joint = np.zeros((pred.size(0),pred.size(1),2))
            for idx, batch in enumerate(pred):
                for idx2, joint in enumerate(batch):
                    joint = joint.detach().cpu()
                    joint = joint.flatten()
                    index = joint.argmax()
                    row = int(index / heatmap_size)
                    col = index % heatmap_size
                    pred_joint[idx,idx2] = np.array([col, row]).flatten()
            pred_joint = torch.tensor(pred_joint)
            pred_joint = pred_joint * multiply ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
            
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not args.projection:
                if iteration == 0 or iteration == int(len(train_dataloader)/2) or iteration == len(train_dataloader) - 1:
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joints, fig, iteration)
                    visualize_prediction(images, pred_joint, fig, 'train', epoch, iteration, args,None)
                    plt.close()

            batch_time.update(time.time() - end)
            end = time.time()
            eta_seconds = batch_time.avg * ((len_total - iteration) + (args.epoch - epoch -1) * len_total)  

            if iteration == len(train_dataloader) - 1:
                logger.info(
                    ' '.join(
                        ['model: {model}', 'dataset_length: {len}','epoch: {ep}', 'iter: {iter}', '/{maxi}, count: {count}/{max_count}']
                    ).format(model=args.model, len=data_len,ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = args.count)
                    + ' 2d_loss: {:.8f}, pck: {:.2f}%, best_loss: {:.8f}, expected_date: {}\n'.format(
                        log_losses.avg,
                        pck, 
                        best_loss,
                        time.ctime(eta_seconds + end))
                )
                writer.add_scalar("Loss/train", log_losses.avg, epoch)

            else:
                logger.info(
                    ' '.join(
                        ['model: {model}', 'dataset_length: {len}','epoch: {ep}', 'iter: {iter}', '/{maxi}, count: {count}/{max_count}']
                    ).format(model=args.model, len=data_len,ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = args.count)
                    + ' 2d_loss: {:.8f}, pck: {:.2f}%, best_loss: {:.8f}\n, expected_date: {}\033[0K\r'.format(
                        log_losses.avg,
                        pck, 
                        best_loss,
                        time.ctime(eta_seconds + end))
                )
            


        return Graphormer_model, optimizer, batch_time

def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss ,logger, writer, batch_time, len_total):

    end = time.time()
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    log_losses = AverageMeter()
    pck_losses = AverageMeter()
    epe_losses = AverageMeter()
    
    if args.model == "ours":
        with torch.no_grad():
            for iteration, (images, gt_2d_joints, heatmap, gt_3d_joints) in enumerate(test_dataloader):
                Graphormer_model.eval()
                batch_size = images.size(0)
                
                images = images.cuda()
                gt_2d_joints = gt_2d_joints
                gt_2d_joint = gt_2d_joints.clone().detach()
                gt_2d_joint = gt_2d_joint.cuda()
                gt_3d_joints = gt_3d_joints.clone().detach()
                gt_3d_joints = torch.tensor(gt_3d_joints).cuda()

                if args.projection: 
                    pred_2d_joints, pred_3d_joints= Graphormer_model(images)
                    pck, threshold = PCK_3d_loss(pred_3d_joints, gt_3d_joints, T= 10)
                    loss = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
                    
                else: 
                    pred_2d_joints= Graphormer_model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); args.loss_3d = 0
                    pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                    pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)
                    # pck= PCK_2d_loss(pred_2d_joints, gt_2d_joint, T= 0.05, threshold = 'proportion')
                    pck = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T= 0.05, threshold = 'proportion')
                    loss = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joint)
                    
                epe_loss, _ = EPE_train(pred_2d_joints, gt_2d_joint)  ## consider invisible joint

                pck_losses.update(pck, batch_size)
                epe_losses.update_p(epe_loss[0], epe_loss[1])
                log_losses.update(loss.item(), batch_size)
                
                if not args.projection:
                    if iteration == 0 or iteration == int(len(test_dataloader)/2) or iteration == len(test_dataloader) - 1:
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joint, fig, iteration)
                        visualize_prediction(images, pred_2d_joints, fig, 'test', epoch, iteration, args,None)
                        plt.close()
                        
                batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = batch_time.avg * ((len(test_dataloader) - iteration) + (args.epoch - epoch -1) *len_total)

                if iteration == len(test_dataloader) - 1:
                    logger.info(
                        ' '.join(
                            ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                        ).format(ep=epoch, iter=iteration, maxi=len(test_dataloader))
                        + ' pck: {:.2f}%, epe: {:.2f}mm, count: {} / {}, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {} \n'.format(
                        # + ' threshold: {} ,pck: {:.2f}%, epe: {:.2f}mm, 2d_loss: {:.2f}, 3d_loss: {:.8f}, count: {} / 50, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {} \n'.format( 
                            pck_losses.avg * 100,
                            epe_losses.avg * 0.26,
                            # log_2d_losses.avg,
                            # log_3d_losses.avg,
                            int(count),
                            args.count,
                            log_losses.avg,
                            best_loss,
                            ctime(eta_seconds + end))
                    )
                    writer.add_scalar("Loss/valid", log_losses.avg, epoch)
                    writer.flush()
                else:
                    logger.info(
                        ' '.join(
                            ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                        ).format(ep=epoch, iter=iteration, maxi=len(test_dataloader))
                         + ' pck: {:.2f}%, epe: {:.2f}mm, count: {} / {}, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                        #  + ' threshold: {} ,pck: {:.2f}%, epe: {:.2f}mm, 2d_loss: {:.2f}, 3d_loss: {:.8f}, count: {} / 50, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                            pck_losses.avg * 100,
                            epe_losses.avg * 0.26,
                            # log_2d_losses.avg,
                            # log_3d_losses.avg,
                            int(count),
                            args.count, 
                            log_losses.avg,
                            best_loss,
                            ctime(eta_seconds + end))
                    )

        return log_losses.avg, count, pck_losses.avg * 100, batch_time

       
    
    else:
        heatmap_size, multiply = 64, 4
        if args.model == "hrnet": heatmap_size, multiply = 128, 2
        for iteration, (images, gt_2d_joints, gt_heatmaps, gt_3d_joints) in enumerate(test_dataloader):
            batch_time = AverageMeter()
            batch_size = images.size(0)
            images = images.cuda()
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()
            gt_heatmaps = gt_heatmaps.cuda()    
             
            pred = Graphormer_model(images)
            
            loss = torch.mean(calc_loss(pred, gt_heatmaps, args))
            log_losses.update(loss.item(), batch_size)
            if args.model == "hourglass": pred = pred[:, -1]

            pred_joint = np.zeros((pred.size(0),pred.size(1),2))
            for idx, batch in enumerate(pred):
                for idx2, joint in enumerate(batch):
                    joint = joint.detach().cpu()
                    joint = joint.flatten()
                    index = joint.argmax()
                    row = int(index / heatmap_size)
                    col = index % heatmap_size
                    pred_joint[idx,idx2] = np.array([col, row]).flatten()
            pred_joint = torch.tensor(pred_joint)
            pred_2d_joints = pred_joint * multiply ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
            
            pck = PCK_2d_loss(pred_2d_joints, gt_2d_joint, T= 0.05, threshold='proportion')
            epe_loss, _ = EPE_train(pred_2d_joints, gt_2d_joint)  ## consider invisible joint
            pck_losses.update(pck, batch_size)
            epe_losses.update_p(epe_loss[0], epe_loss[1])
            
            if not args.projection:
                if iteration == 0 or iteration == int(len(test_dataloader)/2) or iteration == len(test_dataloader) - 1:
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joints, fig, iteration)
                    visualize_prediction(images, pred_2d_joints, fig, 'test', epoch, iteration, args, None)
                    plt.close()

            batch_time.update(time.time() - end)
            end = time.time()
            eta_seconds = batch_time.avg * ((len_total - iteration) + (args.epoch - epoch -1) * len_total)  

            if iteration == len(test_dataloader) - 1:
                logger.info(
                    ' '.join(
                        ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                    ).format(ep=epoch, iter=iteration, maxi=len(test_dataloader))
                    + ' thresold: {} ,pck: {:.2f}%, epe: {:.2f}mm, loss: {:.2f}, count: {} / {}, best_loss: {:.8f}, expected_date: {} \n'.format(
                        threshold,
                        pck_losses.avg * 100,
                        epe_losses.avg * 0.26,
                        log_losses.avg,
                        int(count),
                        args.count,
                        best_loss,
                        ctime(eta_seconds + end))
                )
                writer.add_scalar("Loss/valid", log_losses.avg, epoch)
                writer.flush()
            else:
                logger.info(
                    ' '.join(
                        ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                    ).format(ep=epoch, iter=iteration, maxi=len(test_dataloader))
                    + ' thresold: {} ,pck: {:.2f}%, epe: {:.2f}mm, loss: {:.2f}, count: {} / {}, best_loss: {:.8f}, expected_date: {}'.format(
                        threshold,
                        pck_losses.avg * 100,
                        epe_losses.avg * 0.26,
                        log_losses.avg,
                        int(count),
                        args.count,
                        best_loss,
                        ctime(eta_seconds + end))
                )

        return log_losses.avg, count, pck_losses.avg * 100, batch_time