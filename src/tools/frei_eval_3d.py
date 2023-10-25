import os
import json
import numpy as np
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch.utils import data
from src.tools.models.our_net import get_our_net
from src.utils.argparser import parse_args
from dataset import *
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.visualize import *
from src.utils.dataset_loader import Frei
from tqdm import tqdm


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """Save predictions into a json file."""
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print(
        "Dumped %d joints and %d verts predictions to %s"
        % (len(xyz_pred_list), len(verts_pred_list), pred_out_path)
    )


def main(args):
    n_l = ["src/tools/output/frei/aux_gcn_new"]
    model_list = ["/".join(n.split("/")[2:]) for n in n_l]

    for name in model_list:
        args.name = os.path.join(name, "checkpoint-good/state_dict.bin")
        args.model = args.name.split("/")[1]
        _model = get_our_net(args)
        state_dict = torch.load(args.name)
        _model.load_state_dict(state_dict["model_state_dict"], strict=False)
        _model.cuda()
        pred_name = name.split("/")[-1]
        pred_out_path = os.path.join(name, f"pred_{pred_name}.json")

        test_dataset = make_hand_data_loader(
            args,
            args.val_yaml,
            False,
            is_train=False,
            scale_factor=args.img_scale_factor
        )
        testset_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        
        pbar = tqdm(total=len(testset_loader))
        xyz_list, verts_list = list(), list()

        for idx, (images, _, gt_3d_joints, _) in enumerate(testset_loader):
            _model.eval()
            with torch.no_grad():
                images = images.cuda()
                gt_3d_joints = gt_3d_joints.cuda()
                _, pred_3d_joints, _ = _model(images)
                pred_3d_joints = np.array(pred_3d_joints.cpu())
                for xyz in pred_3d_joints:
                    xyz_list.append(xyz)
                    verts_list.append(np.zeros([778, 3]))
            pbar.update(1)
        pbar.close()
        dump(pred_out_path, xyz_list, verts_list)

        os.system("python eval.py --pred_file_name %s" % pred_out_path)


if __name__ == "__main__":
    args = parse_args(eval=True)
    main(args)
