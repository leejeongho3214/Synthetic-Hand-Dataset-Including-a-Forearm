import os
import json
import numpy as np
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    resume_path = os.path.join('output', args.name)
    _model = get_our_net(args)
    state_dict = torch.load(os.path.join(resume_path, "checkpoint-good/state_dict.bin"))
    _model.load_state_dict(state_dict["model_state_dict"], strict=False)
    _model.cuda()

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
    
    pred_list, gt_list, epe_list = list(), list(), list()
    _model.eval()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    dummy_input = torch.zeros((32, 3, 224, 224)).cuda()

    for _ in range(10):
        _ = _model(dummy_input)
        
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = _model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    
    return
    
    with torch.no_grad():
        for idx, (images, _, gt_3d_joints, _) in enumerate(testset_loader):
            images = images.cuda()
            gt_3d_joints = gt_3d_joints.cuda()
            _, pred_3d_joints, _ = _model(images)
            for idx, pred_joint in enumerate(pred_3d_joints):
                pred_list.append(np.array(pred_joint.detach().cpu()))
                gt_list.append(np.array(gt_3d_joints[idx].detach().cpu()))
                
        for i in range(len(pred_list)):
            aligned_pred = align_w_scale(gt_list[i], pred_list[i])
            gt = np.squeeze(gt_list[i])
            aligned_pred = np.squeeze(aligned_pred)
            diff = gt - aligned_pred
            euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
            epe_list.append(euclidean_dist)
            
    thresholds = np.linspace(0.0, 0.05, 100)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
    pck_curve = list()
    auc_all = list()
    pck_curve_all = list()
    for i in range(21):
        pck_curve = list()
        for t in thresholds:       
            error = np.array(epe_list)[:, i]
            pck = np.mean((error <= t).astype('float'))
            pck_curve.append(pck)
            
        pck_curve = np.array(pck_curve)
        pck_curve_all.append(pck_curve)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        auc_all.append(auc)
        
    auc_all = np.mean(np.array(auc_all))
            
    epe = np.array(epe_list).mean(axis = 0).mean() * 100
    print(f"PA-MPJPE => {epe:.02f} cm, AUC => {auc_all:.02f}")
    
    score_path = 'general_scores.txt'

    if os.path.isfile(score_path):
        mode = "a"
    else:
        mode = "w"
    
    with open(score_path, mode) as fo:
        fo.write("\nname: %s\n" % args.name)
        fo.write('auc=%.3f, xyz_al_mean3d: %.3f cm\n' % (auc_all, epe))
        fo.write("======" * 14)


if __name__ == "__main__":
    args = parse_args(eval=True)
    main(args)
