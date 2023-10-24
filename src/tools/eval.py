from __future__ import print_function, unicode_literals
import os
import sys
sys.path.insert(0, os.path.abspath(
os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.bar import colored
import numpy as np
import json
import argparse
import pip
import matplotlib

import matplotlib.pyplot as plt
import sys
import os

np.set_printoptions(precision=6, suppress=True)


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

# try:
#     import open3d as o3d
# except:
#     install('open3d-python')
#     import open3d as o3d


try:
    from scipy.linalg import orthogonal_procrustes
except:
    install('scipy')
    from scipy.linalg import orthogonal_procrustes


try:
    from utils.fh_utils import *
    from utils.eval_util import EvalUtil

except:
    from freihand.utils.fh_utils import *
    from freihand.utils.eval_util import EvalUtil


# def verts2pcd(verts, color=None):
#     pcd = o3d.PointCloud()
#     pcd.points = o3d.Vector3dVector(verts)
#     if color is not None:
#         if color == 'r':
#             pcd.paint_uniform_color([1, 0.0, 0])
#         if color == 'g':
#             pcd.paint_uniform_color([0, 1.0, 0])
#         if color == 'b':
#             pcd.paint_uniform_color([0, 0, 1.0])
#     return pcd


# def calculate_fscore(gt, pr, th=0.01):
#     gt = verts2pcd(gt)
#     pr = verts2pcd(pr)
#     d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr) # closest dist for each gt point
#     d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt) # closest dist for each pred point
#     if len(d1) and len(d2):
#         recall = float(sum(d < th for d in d2)) / float(len(d2))  # how many of our predicted points lie close to a gt point?
#         precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

#         if recall+precision > 0:
#             fscore = 2 * recall * precision / (recall + precision)
#         else:
#             fscore = 0
#     else:
#         fscore = 0
#         precision = 0
#         recall = 0
#     return fscore, precision, recall


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    curve_data_list = list()
    for item in curve_list:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path_path.png")
        plt.savefig(img_path, bbox_inches=0, dpi=300)

        # write image and create html embedding
        data_uri1 = open(img_path, 'rb').read().encode(
            'base64').replace('\n', '')
        img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
        curve_data_list.append((item.text, img_tag1))

        os.remove(img_path)

    htmlString = '''<!DOCTYPE html>
    <html>
    <body>
    <h1>Detailed results:</h1>'''

    for i, (text, img_embed) in enumerate(curve_data_list):
        htmlString += '''
        <h2>%s</h2>
        <p>
        <img border="0" %s alt="FROC" width="576pt" height="432pt">
        </p>
        <p>Raw curve data:</p>
        
        <p>x_axis: <small>%s</small></p>
        <p>y_axis: <small>%s</small></p>
        
        ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)

    htmlString += '''
    </body>
    </html>'''

    htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    htmlfile.write(htmlString)
    htmlfile.close()


def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x)
             for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception(
            'Giving up, because its not clear which file to evaluate.')


def main(gt_path, pred_path, output_dir, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'

    # load eval annotations
    xyz_list, verts_list = json_load(os.path.join(gt_path, '%s_xyz.json' % set_name)), json_load(
        os.path.join(gt_path, '%s_verts.json' % set_name))

    # load predicted values

    # pred_file = _search_pred_file(pred_path, pred_file_name)
    pred_file = pred_file_name
    print('Loading predictions from %s' % pred_file)
    with open(pred_file, 'r') as fi:
        pred = json.load(fi)

    assert len(pred) == 2, 'Expected format mismatch.'
    assert len(pred[0]) == len(xyz_list), 'Expected format mismatch.'
    assert len(pred[1]) == len(xyz_list), 'Expected format mismatch.'

    # init eval utils
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(
        num_kp=778), EvalUtil(num_kp=778)

    shape_is_mano = None

    try:
        from tqdm import tqdm
        rng = tqdm(range(db_size(set_name)))
    except:
        rng = range(db_size(set_name))

    # iterate over the dataset once
    My_list = []
    for idx in rng:

        if idx >= db_size(set_name):
            break

        xyz, verts = xyz_list[idx], verts_list[idx]
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred, verts_pred = pred[0][idx], pred[1][idx]
        xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )

        if shape_is_mano is None:
            if verts_pred.shape[0] == verts.shape[0]:
                shape_is_mano = True
            else:
                shape_is_mano = False

        if shape_is_mano:
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        My_list.append([xyz.tolist(), xyz_pred_aligned.tolist()])
        if shape_is_mano:
            verts_pred_aligned = align_w_scale(verts, verts_pred)
        else:
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
            verts_pred_aligned = align_by_trafo(verts_pred, trafo)

        # Aligned errors
        eval_xyz_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_aligned
        )

        if shape_is_mano:
            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned
            )

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al, pck_list = eval_xyz_aligned.get_measures(
        0.0, 0.05, 100)
    want = np.array(pck_list).mean(axis=0)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' %
          (xyz_al_auc3d, xyz_al_mean3d * 100.0))

    with open('mesh.json', 'w') as f:
        json.dump(list(want), f)
    
    name_list = pred_file.split('/')
    score_path = os.path.join("/".join(name_list[:2]), f'general_scores.txt')

    if os.path.isfile(score_path):
        mode = "a"
    else:
        mode = "w"

    with open(score_path, mode) as fo:
        xyz_al_mean3d *= 100
        fo.write("\nname: %s\n" % "/".join(name_list[:-1]))
        fo.write('auc=%.3f, xyz_al_mean3d: %.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d))
        fo.write("======" * 14)
    print(colored("Writting => %s" % score_path, "red"))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show some samples from the dataset.')
    parser.add_argument('--input_dir', type=str, default="../../datasets/frei_test", required=False,
                        help='Path to where prediction the submited result and the ground truth is.')
    parser.add_argument('--output_dir', type=str, default="../../freihand", required=False,
                        help='Path to where the eval result should be.')
    parser.add_argument('--pred_file_name', type=str,
                        help='Name of the eval file.')
    args = parser.parse_args()

    # call eval
    main(
        args.input_dir,
        "",
        args.output_dir,
        args.pred_file_name,
        set_name='evaluation'
    )

# import pandas as pd

# new_index = np.arange(0, 50, 0.5)
# df = pd.DataFrame(want, index = new_index)
# df.index.stop = 50
# plt.figure(figsize=(15, 8))  # 그래프의 크기 설정 (선택사항)
# plt.scatter(df.index, df, label='Data', s=10, color='blue', marker='o')  # s는 점의 크기

# plt.plot(df, linewidth=1, label='Line')
# plt.grid(True)
# plt.title('3D PCK on pose(FreiHAND)')
# plt.xlabel('error (mm)')
# plt.ylabel('3D PCK of joint')

# # 범례 추가
# plt.legend(loc='best', labels=['MeshGraphormer(ICCV2021): AUC=0.874', 'Ours: AUC=0.865'])

# plt.savefig('a')