import sys
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
import argparse
import torchvision.models as models
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.tools.dataset import save_checkpoint
from src.utils.comm import get_rank
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir
import torch
import os
import gc
import json
import os
from src.datasets.build import make_hand_data_loader
import time
import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torch
from loss import keypoint_2d_loss, calcu_one, adjust_learning_rate, keypoint_3d_loss, PCK_2d_loss, PCK_2d_loss_HIU
from src.utils.geometric_layers import camera_calibration, orthographic_projection
from src.utils.metric_logger import AverageMeter
from visualize import visualize_prediction, visualize_gt
import sys

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--multiscale_inference", default=False, action='store_true', )
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument("--aml_eval", default=False, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X steps.")
    parser.add_argument("--resume_path", default='HIU', type=str)
    #############################################################################################
            ## Set hyper parameter ##
    #############################################################################################
    parser.add_argument("--loss_2d", default=1, type=float,)
    parser.add_argument("--loss_3d", default=1, type=float,
                        help = "it is weight of 3d regression and '0' mean only 2d joint regression")
    parser.add_argument("--train", default='train', type=str, choices=['pre-train, train, fine-tuning'],
                        help = "3 type train method")
    parser.add_argument("--name", default='HIU_DMTL_full',
                        help = '20k means CISLAB 20,000 images',type=str)
    parser.add_argument("--root_path", default=f'output', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--output_path", default='HIU', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--iter", action='store_true')
    parser.add_argument("--resume", action='store_true')
    #############################################################################################

    #############################################################################################
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

def load_model(args):
    epo = 0
    best_loss = 0

    # args.output_dir = op.join(args.output_dir, f'{args.train_data}_2d:{args.loss_2d}_3d:{args.loss_3d}')
    # if os.path.isdir(op.join(args.output_dir, 'checkpoint-good')) == True:
    #     args.resume_checkpoint = op.join(args.output_dir, 'checkpoint-good/state_dict.bin')

    if args.iter == True:
        args.resume_checkpoint = os.path.join(os.path.join(args.root_path, args.output_path),'checkpoint-iter/state_dict.bin')
    else:
        args.resume_checkpoint = os.path.join(os.path.join(args.root_path, args.output_path),'checkpoint-good/state_dict.bin')
    args.output_dir = os.path.join(args.root_path, args.output_path)

    if args.resume == False:
        args.resume_checkpoint = 'None'
    if os.path.isdir(args.output_dir) == False:
        mkdir(args.output_dir)

    logger = setup_logger(args.name, args.output_dir, get_rank())
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    trans_encoder = []
    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

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
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch == 'hrnet-w64':
        hrnet_yaml = '../../models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = '../../models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        # logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info('Graphormer encoders total parameters: {}\n'.format(total_params))
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    logger.info('Backbone total parameters: {}\n'.format(backbone_total_params))

    # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
    _model = Graphormer_Network(args, config, backbone, trans_encoder, token = 70)

    if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
        state_dict = torch.load(args.resume_checkpoint)
        best_loss = state_dict['best_loss']
        epo = state_dict['epoch']
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)
        logger.info("Resume: Loading from checkpoint {}\n".format(args.resume_checkpoint))
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    _model.to(args.device)
    return _model, logger, best_loss, epo

def train(args, train_dataloader, Graphormer_model, epoch, best_loss, data_len ,logger):
    gc.collect()
    torch.cuda.empty_cache()
    max_iter = len(train_dataloader)
    optimizer = torch.optim.Adam(params=list(Graphormer_model.parameters()),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_3d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )


    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    # log_loss_camera = AverageMeter()
    logger.info(f'loss_2d = {args.loss_2d} // loss_3d = {args.loss_3d}\n')
    for iteration, (images, gt_2d_joints, gt_3d_joints) in enumerate(train_dataloader):
        batch_time = AverageMeter()
        batch_inference_time = time.time()
        Graphormer_model.train()
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        images = images.cuda()
        gt_2d_joints = gt_2d_joints/224
        gt_2d_joint = gt_2d_joints.clone().detach()
        gt_2d_joint = gt_2d_joint.cuda()
        gt_3d_joints = gt_3d_joints.cuda()
        # gt_camera = camera.cuda()

        pred_camera, pred_3d_joints = Graphormer_model(images)

        if args.loss_3d == 0:
            pred_2d_joints = pred_3d_joints[:,:,:-1]
            loss_3d_joints = 0
        else:
            # pred_2d_joints, rot = camera_calibration(pred_3d_joints.contiguous(), pred_camera.contiguous())
            pred_2d_joints = orthographic_projection(pred_3d_joints.contiguous(), pred_camera.contiguous())
            loss_3d_joints = keypoint_3d_loss(criterion_3d_keypoints, pred_3d_joints, gt_3d_joints)

        pred_2d_joints = pred_2d_joints/224
        # pred_camera = torch.concat([rot.cuda(), pred_camera[:,3:]],1)
        # loss_camera = criterion_2d_keypoints(pred_camera, gt_camera).mean()
        loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joint)
        # loss = args.loss_2d * loss_2d_joints + args.loss_3d * loss_3d_joints + 0.01 * loss_camera
        loss = args.loss_2d * loss_2d_joints + args.loss_3d * loss_3d_joints
        # loss = args.loss_2d * loss_2d_joints
        log_loss_2djoints.update(loss_2d_joints, batch_size)
        log_loss_3djoints.update(loss_3d_joints, batch_size)
        # log_loss_camera.update(loss_camera, batch_size)
        log_losses.update(loss.item(), batch_size)

       # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - batch_inference_time, n=1)
        if iteration % 1000 == 800:
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, 'iter_2', iteration=iteration, logger=logger)

        elif iteration % 1000 == 500:
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, 'iter', iteration=iteration, logger=logger)
            
        if args.visualize == True:
            # if iteration % 10000 == 1:
            fig = plt.figure()
            visualize_gt(images, gt_2d_joint * 224, fig)
            visualize_prediction(images, pred_2d_joints * 224, fig)
            plt.close()

        if iteration == len(train_dataloader) - 1:
            logger.info(
                ' '.join(
                    ['dataset_length: {len}','epoch: {ep}', 'iter: {iter}', '/{maxi}']
                ).format(len=data_len,ep=epoch, iter=iteration, maxi=max_iter)
                # + ' 2d_loss: {:.6f}, 3d_loss: {:.6f}, camera: {:.6f}, toatl_loss: {:.6f}, best_pck: {:.2f} %, lr: {:.6f}, time: {:.6f}\n'.format(
                + ' 2d_loss: {:.6f}, 3d_loss: {:.6f},  toatl_loss: {:.6f}, best_pck: {:.2f} %, lr: {:.6f}, time: {:.6f}\n'.format(
                    log_loss_2djoints.avg,
                    log_loss_3djoints.avg,
                    # log_loss_camera.avg,
                    log_losses.avg,
                    best_loss*100,
                    optimizer.param_groups[0]['lr'],
                    batch_time.avg)
            )

        else:
            logger.info(
                ' '.join(
                    ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi}']
                ).format(len=data_len,ep=epoch, iter=iteration, maxi=max_iter)
                + ' 2d_loss: {:.6f}, 3d_loss: {:.6f},  toatl_loss: {:.6f}, best_pck: {:.2f} %, lr: {:.6f}, time: {:.6f}'.format(
                    log_loss_2djoints.avg,
                    log_loss_3djoints.avg,
                    # log_loss_camera.avg,zc
                    log_losses.avg,
                    best_loss * 100,
                    optimizer.param_groups[0]['lr'],
                    batch_time.avg)
            )

        if iteration % 5000 == 4999:
            gc.collect()
            torch.cuda.empty_cache()


    return Graphormer_model, optimizer

def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss ,logger):

    max_iter = len(test_dataloader)
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    pck_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    with torch.no_grad():
        
        for iteration, (images, gt_2d_joints, _) in enumerate(test_dataloader):

            Graphormer_model.eval()
            batch_size = images.size(0)
            
            images = images.cuda()
            gt_2d_joints = gt_2d_joints
            gt_2d_joint = gt_2d_joints.clone().detach()
            gt_2d_joint = gt_2d_joint.cuda()

            pred_camera, pred_3d_joints = Graphormer_model(images)
            if args.loss_3d == 0:
                pred_2d_joints = pred_3d_joints[:, :, :-1]
            else:
                pred_2d_joints = camera_calibration(pred_3d_joints.contiguous(), pred_camera.contiguous())

            pred_2d_joints = pred_2d_joints * 224
            correct, visible_point = PCK_2d_loss_HIU(pred_2d_joints, gt_2d_joint, images)
            pck_losses.update_p(correct, visible_point)

            # if args.visualize == True:
            if iteration % 1000 == 1:
        # if iteration == 0:
                fig = plt.figure()
                visualize_gt(images, gt_2d_joint, fig)
                visualize_prediction(images, pred_2d_joints, fig, epoch,iteration)
                plt.close()

            if iteration == len(test_dataloader) - 1:
                logger.info(
                    ' '.join(
                        ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                    ).format(ep=epoch, iter=iteration, maxi=max_iter)
                    + ' pck: {:.2f}%, 3d: {},  count: {} / 50,best_pck: {:.2f} %\n'.format(
                        pck_losses.avg * 100,
                        # log_loss_3djoints.avg,
                        0,
                        int(count),
                        best_loss*100)
                )
            else:
                logger.info(
                    ' '.join(
                        ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                    ).format(ep=epoch, iter=iteration, maxi=max_iter)
                    + '  pck: {:.2f}%, 3d: {}, count: {} / 50, best_pck: {:.2f} %'.format(
                        pck_losses.avg * 100,
                        # log_loss_3djoints.avg,
                        0,
                        int(count),
                        best_loss*100)
                )


    del test_dataloader
    gc.collect()

    return pck_losses.avg