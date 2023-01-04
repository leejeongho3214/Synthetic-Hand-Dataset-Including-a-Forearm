# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import sys

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
def reset_txt(txt_name):
    if os.path.isfile(txt_name):
        os.remove(txt_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
def resume_checkpoint(args, _model):
    state_dict = torch.load(args.resume_checkpoint)
    best_loss = state_dict['best_loss']
    epoch = state_dict['epoch']
    count = state_dict['count']
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    del state_dict
    
    return best_loss, epoch, _model, count