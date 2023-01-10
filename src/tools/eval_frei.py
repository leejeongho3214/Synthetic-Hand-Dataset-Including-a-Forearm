import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from torch.utils import data
torch.device('cuda')
from argparser import load_model, parse_args
import sys
from dataset import *
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.visualize import *
from src.utils.dataset_loader import Frei
from tqdm import tqdm    

def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))

def main(args):
    name = "output/ours/general/frei_3d_mid"
    args.name = os.path.join(name, "checkpoint-good/state_dict.bin")
    args.model = args.name.split('/')[1]
    if args.model == "other_dataset": args.model = "ours"
    args.projection = True
    _model, _, _,_ = load_model(args)
    state_dict = torch.load(args.name)
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    _model.cuda()
    pred_out = "../../freihand"
    pred_name = name.split("/")[-1]
    pred_out_path = os.path.join(pred_out, f"pred_{pred_name}.json")
    
    test_dataset = Frei(args)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


    pbar = tqdm(total = len(testset_loader)) 
    xyz_list, verts_list = list(), list()
    for idx, (images, _, _, gt_3d_joints) in enumerate(testset_loader):

        _model.eval()
        with torch.no_grad():
            images = images.cuda()
            gt_3d_joints = gt_3d_joints.cuda()
            _, pred_3d_joints = _model(images)
            pred_3d_joints = np.array(pred_3d_joints.cpu())
            for xyz in pred_3d_joints:
                xyz_list.append(xyz)
                verts_list.append(np.zeros([778, 3]))

        pbar.update(1)
    pbar.close()
    dump(pred_out_path, xyz_list, verts_list)
    os.system(f"python ../../freihand/eval.py --pred_file_name ../../freihand/{pred_out_path.split('/')[-1]}")
            

if __name__ == "__main__":
    args, logger = parse_args()
    main(args)
    

