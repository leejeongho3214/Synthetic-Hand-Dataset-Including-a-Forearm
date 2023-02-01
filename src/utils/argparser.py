
import os
import argparse
from src.tools.models.our_net import get_our_net
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("name", default='None',
                        help = 'You write down to store the directory path',type=str)
    parser.add_argument("--dataset", default='ours', type=str, required=False)
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--ratio_of_our", default=1, type=float,
                        help="Our dataset have 420k imaegs so you can use train data as many as you want, according to this ratio")
    parser.add_argument("--ratio_of_aug", default=0.6, type=float,
                        help="You can use color jitter to train data as many as you want, according to this ratio")
    parser.add_argument("--ratio_of_add", default=0.1, type=float,
                        help = "set ratio that how many you add our dataset")
    parser.add_argument("--epoch", default=50, type=int) 
    parser.add_argument("--loss_2d", default=0, type=float)
    parser.add_argument("--loss_3d", default=0, type=float)
    parser.add_argument("--loss_3d_mid", default=0, type=float)
    parser.add_argument("--set", default=None, type=str,
                        help = "when input set, the existing output folder is removed as the initialize")
    parser.add_argument("--center", action='store_true')
    parser.add_argument("--nn", action='store_true')
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--scale", action='store_true')
    parser.add_argument("--plt", action='store_true')
    parser.add_argument("--logger", action='store_true')
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--batch_size", default=32, type=int)
    
    args = parser.parse_args()
    args, logger = pre_arg(args)
    args.logger = logger
    
    return args


def load_model(args):
    epoch = 0
    best_loss = np.inf
    count = 0
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.device = torch.device(args.device)

    _model = get_our_net(args) ## output: 21 x 2
        
    log_dir = f'tensorboard/{args.name}'
    writer = SummaryWriter(log_dir)
    
    if args.name.split("/")[0] != "final_model":
        if args.reset: 
            if os.path.isfile(os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin')):
                if True if input("There is resume_point but do you want to delete?") == "o" else False:
                    reset_folder(log_dir); reset_folder(os.path.join(args.root_path, args.name)); 
                    print(colored("Ignore the check-point model", "green"))
                    args.reset = "resume but init"
            else:
                args.reset = "init"
        else: 
            if os.path.isfile(os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin')):
                best_loss, epoch, _model, count = resume_checkpoint(_model, os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin'))
                args.logger.debug("Loading ===> %s" % os.path.join(args.root_path, args.name))
                print(colored("Loading ===> %s" % os.path.join(args.root_path, args.name), "green"))
                args.reset = "resume"
            else:
                reset_folder(log_dir); args.reset = "init"
        
    _model.to(args.device)
    
    return _model, best_loss, epoch, count, writer



def train(args, train_dataloader, test_dataloader, Graphormer_model, epoch, best_loss, data_len ,count, writer, pck, len_total, batch_time):
    end = time.time()
    phase = 'TRAIN'
    runner = Runner(args, Graphormer_model, epoch, train_dataloader, test_dataloader, phase, batch_time,  data_len, len_total, count, pck, best_loss, writer)

    Graphormer_model, optimizer, batch_time= runner.our(end)
        
    return Graphormer_model, optimizer, batch_time, best_loss

def valid(args, train_dataloader, test_dataloader, Graphormer_model, epoch, count, best_loss,  data_len , writer, batch_time, len_total, pck):
    end = time.time()
    phase = 'VALID'
    runner = Runner(args, Graphormer_model, epoch, train_dataloader, test_dataloader, phase, batch_time,  data_len, len_total, count, pck, best_loss, writer)

    loss, count, pck, batch_time = runner.our(end)
       
    return loss, count, pck, batch_time

