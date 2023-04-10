
import argparse
import os
from src.utils.bar import colored
from src.utils.comm import get_rank
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir
from src.utils.dir import reset_file

def pre_arg(args, eval):
    output_dir = os.path.join('output', args.name)
    if args.reset or not os.path.isfile(os.path.join(output_dir,'checkpoint-good/state_dict.bin')): reset_file(os.path.join(output_dir, "log.txt"))
    if not output_dir.split('/')[1] == "output" and not os.path.isfile((output_dir)):  mkdir(output_dir); logger = setup_logger(args.name, output_dir, get_rank())
    else: logger = None
    logger.debug(args)
    if not eval: print(colored(args, "yellow"))
    args.root_path = 'output'
    args.output_dir = os.path.join(args.root_path, args.name)
    args.logger = logger
    args.num_train_epochs = int(50)
    args.multiscale_inference = False
    args.sc = float(1.0)
    args.aml_eval = False
    args.logging_steps = int(100)
    args.lr = float(1e-4)
    args.vertices_loss_weight = float(1.0)
    args.joints_loss_weight = float(1.0)
    args.vloss_w_full = float(0.5)
    args.vloss_w_sub = float(0.5)
    args.drop_out = float(0.1)
    args.num_workers = int(4)
    args.img_scale_factor = int(1)
    
    args.image_file_or_path = str('../../samples/unity/images/train/Capture0')
    args.train_yaml = str('../../datasets/freihand/train.yaml')
    args.val_yaml = str('../../datasets/freihand/test.yaml')
    args.data_dir = str('datasets')
    args.model_name_or_path = str('../modeling/bert/bert-base-uncased/')
    args.config_name = str("")
    args.arch = "hrnet-w64"
    args.hidden_size = int(-1) 
    args.num_attention_heads = int(4)
    args.intermediate_size = int(-1)
    args.input_feat_dim = [2048,512,128]
    args.hidden_feat_dim = [1024,256,64]

    args.mesh_type = str('hand')
    args.run_eval_only = True
    args.device = str('cuda')
    args.seed = int(88)




    
    return args, logger