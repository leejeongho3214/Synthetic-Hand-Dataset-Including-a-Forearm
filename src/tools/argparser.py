
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
from src.utils.method import Runner
from src.utils.dir import  resume_checkpoint
import numpy as np
from matplotlib import pyplot as plt
import torch
from src.utils.loss import *
from src.utils.geometric_layers import *
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
    end = time.time()
    phase = 'TRAIN'
    runner = Runner(args, Graphormer_model, epoch, train_dataloader, phase)
    if args.model == "ours":
        Graphormer_model, optimizer, batch_time = runner.our(train_dataloader,end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase)
    else:
        Graphormer_model, optimizer, batch_time = runner.other(train_dataloader,end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase)
    return Graphormer_model, optimizer, batch_time

def test(args, test_dataloader, Graphormer_model, epoch, count, best_loss,  data_len ,logger, writer, batch_time, len_total, pck):
    end = time.time()
    phase = 'VALID'
    runner = Runner(args, Graphormer_model, epoch, test_dataloader, phase)
    if args.model == "ours":
        loss, count, pck, batch_time = runner.our(test_dataloader, end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase)
    else:
       loss, count, pck, batch_time = runner.other(test_dataloader, end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase)
    return loss, count, pck, batch_time
