
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modeling.hrnet.config.default import update_config
from src.modeling.hrnet.config.default import _C as cfg
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_pose_net as get_cls_net_gridfeat
from src.utils.pre_argparser import pre_arg

from src.tools.models.our_net import get_our_net

from src.modeling.simplebaseline.config import config as config_simple
from src.modeling.simplebaseline.pose_resnet import get_pose_net
import torch
import os
import time
from src.utils.dir import  resume_checkpoint
import numpy as np
from matplotlib import pyplot as plt
import torch
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.metric_logger import AverageMeter
from src.utils.visualize import *
from time import ctime
from src.modeling.hourglass.posenet import PoseNet

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default='None',
                        help = 'You write down to store the directory path',type=str)
    parser.add_argument("--root_path", default=f'output', type=str, required=False,
                        help="The root directory to save location which you want")
    parser.add_argument("--model", default='ours', type=str, required=False,
                        help="you can choose model like hrnet, simplebaseline, hourglass, ours")
    parser.add_argument("--dataset", default='ours', type=str, required=False,
                        help="you can choose dataset like ours, coco, interhand, rhd, frei, hiu, etc.")

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--ratio_of_our", default=0.3, type=float,
                        help="Our dataset have 420k imaegs so you can use train data as many as you want, according to this ratio")
    parser.add_argument("--ratio_of_other", default=0.3, type=float)
    parser.add_argument("--ratio_of_aug", default=0.2, type=float,
                        help="You can use color jitter to train data as many as you want, according to this ratio")
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--loss_2d", default=0, type=float)
    parser.add_argument("--loss_3d", default=1, type=float)
    parser.add_argument("--loss_3d_mid", default=0, type=float)
    parser.add_argument("--scale", action='store_true',
                        help = "If you write down, The 3D joint coordinate would be normalized according to distance between 9-10 keypoint")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--color", action='store_true',
                        help="If you write down, This dataset would be applied color jitter to train data, according to ratio of aug")
    parser.add_argument("--general", action='store_true', 
                        help="If you write down, This dataset would be view of the general")
    parser.add_argument("--projection", action='store_true',
                        help="If you write down, The output of model would be 3d joint coordinate")
    
    args = parser.parse_args()
    args, logger = pre_arg(args)
    
    return args, logger


def load_model(args):
    epoch = 0
    best_loss = np.inf
    count = 0

    if not args.resume: args.resume_checkpoint = 'None'
    else: args.resume_checkpoint = os.path.join(os.path.join(args.root_path, args.name),'checkpoint-good/state_dict.bin')

        
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
        _model = get_our_net(args) ## output: 21 x 2

    if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
        best_loss, epoch, _model, count = resume_checkpoint(args, _model)
        
    _model.to(args.device)
    return _model, best_loss, epoch, count



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
            
            parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
            gt_3d_mid_joints = torch.ones(batch_size, 20, 3)
            for i in range(20):
                gt_3d_mid_joints[:, i, :] =  (gt_3d_joints[:, i + 1, :] + gt_3d_joints[:, parents[i + 1], :]) / 2
            
            if args.scale:
                scale = ((gt_3d_joints[:, 10,:] - gt_3d_joints[:, 9, :])**2).sum(-1).sqrt()
                for i in range(batch_size):
                    gt_3d_joints[i] = gt_3d_joints[i]/scale[i]

            images = images.cuda()
            
            if args.projection: pred_2d_joints, pred_3d_joints= Graphormer_model(images)
            else: pred_2d_joints= Graphormer_model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); args.loss_3d = 0

            pred_3d_mid_joints = torch.ones(batch_size, 20, 3)
            for i in range(20):
                pred_3d_mid_joints[:, i, :] =  (pred_3d_joints[:, i + 1, :] + pred_3d_joints[:, parents[i + 1], :]) / 2
            
            loss_2d= keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joint)
            loss_3d = keypoint_3d_loss(criterion_keypoints, pred_3d_mid_joints, gt_3d_mid_joints)
            loss_3d_re = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
            loss_3d_mid = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
            if args.projection:
                loss = args.loss_2d * loss_2d + args.loss_3d * loss_3d + args.loss_3d_mid * loss_3d_mid
            else:
                loss = loss_2d
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
                    visualize_pred(images, pred_2d_joints, fig, 'train', epoch, iteration, args, None)
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
                if iteration % args.logging_steps == 0:
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
                else:
                    logger.debug(
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
                    visualize_pred(images, pred_joint, fig, 'train', epoch, iteration, args,None)
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
                if iteration % 100 == 0:
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
                else:
                    logger.debug(
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
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    log_losses = AverageMeter()
    pck_losses = AverageMeter()
    epe_losses = AverageMeter()
    
    if args.model == "ours":
        with torch.no_grad():
            for iteration, (images, gt_2d_joints, _, gt_3d_joints) in enumerate(test_dataloader):
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
                    pck, threshold = PCK_3d_loss(pred_3d_joints, gt_3d_joints, T= 1)
                    loss = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
                    # loss = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
                    
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
                        visualize_pred(images, pred_2d_joints, fig, 'test', epoch, iteration, args,None)
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
                    logger.debug(
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
                    visualize_pred(images, pred_2d_joints, fig, 'test', epoch, iteration, args, None)
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
                logger.debug(
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