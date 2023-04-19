import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2" 
import numpy as np
import torch
from torch.utils import data
torch.device('cuda')
from src.tools.models.our_net import get_our_net
from src.utils.argparser import parse_args
import sys
from dataset import *
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.visualize import *
from src.utils.dataset_loader import Frei
from tqdm import tqdm    


def dump(pred_out_path, xyz_pred_list, gt_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                gt_list
                
            ], fo)
    print('Dumped %d joints to %s' % (len(xyz_pred_list),  pred_out_path))

def main(args):
    root = 'output/ours'
    n_l  = ["frei/gcn/hrnet/loss/heatmap/add/gcn_0_0_1_layer_2"]
    model_list = [os.path.join(root, n) for n in n_l]
    
    # model_path = "output/ours/our_part"
    # model_list = list()
    # for (root, _, files) in os.walk(model_path):
    #     for file in files:
    #         if '.bin' in file:
    #             model_list.append('/'.join(root.split('/')[:-1]))
                
    for name in model_list:
        # name = "output/ours/dart/3d"
        args.name = os.path.join(name, "checkpoint-good/state_dict.bin")
        args.model = args.name.split('/')[1]
        if args.model == "other_dataset": args.model = "ours"
        _model = get_our_net(args)
        state_dict = torch.load(args.name)
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)
        _model.cuda()
        pred_name = name.split("/")[-1]
        pred_out_path = os.path.join(name, f"pred_{pred_name}.json")
        
        test_dataset = Frei(args)
        testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        pbar = tqdm(total = len(testset_loader)) 
        xyz_list, verts_list, gt_list = list(), list(), list()
        
        for images, _,  gt_3d_joints in testset_loader:
            _model.eval()
            with torch.no_grad():
                images = images.cuda()
                _, pred_3d_joints, _, _ = _model(images)
                pred_3d_joints = np.array(pred_3d_joints.cpu())
                for idx, xyz in enumerate(pred_3d_joints):
                    xyz_list.append(xyz)
                    verts_list.append(np.zeros([778, 3]))
                    gt_list.append(gt_3d_joints[idx].tolist())
            pbar.update(1)
        pbar.close()
        dump(pred_out_path, xyz_list, gt_list)
            
        os.system("python eval.py --pred_file_name %s" %pred_out_path)
            

if __name__ == "__main__":
    args= parse_args(eval = True)
    main(args)

