import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.miscellaneous import mkdir
from src.utils.dir import reset_folder


def visualize_pred(
    images, pred_2d_joint, fig, method=None, epoch=0, iteration=0, args=None
):
    for num in range(16):
        image = visualize(images, pred_2d_joint, num)
        ax1 = fig.add_subplot(6, 6, (num + 1) * 2)

        ax1.imshow(image)
        ax1.axis("off")

    if method == "evaluation":
        if not os.path.isdir("eval_image"):
            mkdir("eval_image")
        plt.savefig(os.path.join("eval_image", f"iter_{iteration}.jpg"), dpi=2000)

    elif method == "train":
        root = f"{args.output_dir}/train_image"
        epoch_path = os.path.join(root, f"{epoch}_epoch")
        if iteration == 0 and epoch == 0:
            reset_folder(root)
            reset_folder("/".join(root.split("/")[:-1]) + "/test_image")
        if not os.path.isdir(epoch_path):
            mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f"iter_{iteration}.jpg"), dpi=2000)

    elif method == "test":
        epoch_path = f"{args.output_dir}/test_image/{epoch}_epoch"
        if not os.path.isdir(epoch_path):
            mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f"iter_{iteration}.jpg"), dpi=2000)

    else:
        assert False, "method is the wrong name"


def visualize(images, joint_2d, num):
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array(
        [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    )
    for i in range(21):
        cv2.circle(
            image,
            (int(joint_2d[num][i][0]), int(joint_2d[num][i][1])),
            2,
            [0, 1, 0],
            thickness=-1,
        )
        if i != 0:
            cv2.line(
                image,
                (int(joint_2d[num][i][0]), int(joint_2d[num][i][1])),
                (int(joint_2d[num][parents[i]][0]), int(joint_2d[num][parents[i]][1])),
                [0, 0, 1],
                1,
            )

    return image


def visualize_only_gt(
    images, gt_2d_joint, fig, method=None, epoch=0, iteration=0, args=None
):
    for num in range(min(len(images), 16)):
        image = visualize(images, gt_2d_joint, num)
        ax1 = fig.add_subplot(4, 4, num + 1)
        ax1.imshow(image)
        ax1.axis("off")

    if method == "train":
        root = f"{args.output_dir}/train_image"
    else:
        root = f"{args.output_dir}/val_image"

    epoch_path = os.path.join(root, f"{epoch}_epoch")
    if iteration == 0 and epoch == 0:
        reset_folder(root)
    if not os.path.isdir(epoch_path):
        mkdir(epoch_path)
    plt.savefig(os.path.join(epoch_path, f"iter_{iteration}.jpg"), dpi=2000)


def visualize_3d(
    images, gt_2d_joint, gt_3d_joint, pred_3d_joint, method, epoch, iteration, args
):
    fig = plt.figure(figsize=(12, 12))
    gt_2d_joint = gt_2d_joint.detach().cpu()
    gt_3d_joint = gt_3d_joint.detach().cpu()
    pred_3d_joint = pred_3d_joint.detach().cpu()

    num = 5
    for idx in range(1, num):
        image = np.moveaxis(images[idx].detach().cpu().numpy(), 0, -1)
        image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
        
        ax = fig.add_subplot(num, 4, 4 * (idx - 1) + 1)
        ax.imshow(image)
        visualize_joints_2d(ax, gt_2d_joint[idx], joint_idxs=False, alpha=0.5)
        ax.axis("off")

        ax = fig.add_subplot(num, 4, 4 * (idx - 1) + 2)
        visualize_joints_2d(
            ax,
            np.stack([gt_3d_joint[idx][:, 0], gt_3d_joint[idx][:, 1]], axis=1),
            joint_idxs=False,
        )
        visualize_joints_2d(
            ax,
            np.stack([pred_3d_joint[idx][:, 0], pred_3d_joint[idx][:, 1]], axis=1),
            alpha=0.2,
            joint_idxs=False,
        )

        ax = fig.add_subplot(num, 4, 4 * (idx - 1) + 3)
        visualize_joints_2d(
            ax,
            np.stack([gt_3d_joint[idx][:, 1], gt_3d_joint[idx][:, 2]], axis=1),
            joint_idxs=False,
        )
        visualize_joints_2d(
            ax,
            np.stack([pred_3d_joint[idx][:, 1], pred_3d_joint[idx][:, 2]], axis=1),
            alpha=0.2,
            joint_idxs=False,
        )

        ax = fig.add_subplot(num, 4, 4 * (idx - 1) + 4)
        visualize_joints_2d(
            ax,
            np.stack([gt_3d_joint[idx][:, 0], gt_3d_joint[idx][:, 2]], axis=1),
            joint_idxs=False,
        )
        visualize_joints_2d(
            ax,
            np.stack([pred_3d_joint[idx][:, 0], pred_3d_joint[idx][:, 2]], axis=1),
            alpha=0.2,
            joint_idxs=False,
        )

    if method == "evaluation":
        if not os.path.isdir("eval_image"):
            mkdir("eval_image")
        plt.savefig(os.path.join("eval_image", f"iter_{iteration}_3d.jpg"))

    elif method == "train":
        root = f"{args.output_dir}/train_image"
        epoch_path = os.path.join(root, f"{epoch}_epoch")
        if iteration == 0 and epoch == 0:
            reset_folder(root)
            reset_folder("/".join(root.split("/")[:-1]) + "/test_image")
        if not os.path.isdir(epoch_path):
            mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f"iter_{iteration}_3d.jpg"))

    elif method == "test":
        epoch_path = f"{args.output_dir}/test_image/{epoch}_epoch"
        if not os.path.isdir(epoch_path):
            mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f"iter_{iteration}_3d.jpg"))

    else:
        assert False, "method is the wrong name"


def visualize_jupyter(image, joint_2d):
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array(
        [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    )
    for i in range(21):
        cv2.circle(
            image,
            (int(joint_2d[i][0]), int(joint_2d[i][1])),
            2,
            [0, 1, 0],
            thickness=-1,
        )
        if i != 0:
            cv2.line(
                image,
                (int(joint_2d[i][0]), int(joint_2d[i][1])),
                (int(joint_2d[parents[i]][0]), int(joint_2d[parents[i]][1])),
                [0, 0, 1],
                1,
            )

    return image


def add_scatter_proj(ax, pred_objpoints3d, proj="z"):
    proj_1, proj_2 = get_proj_axis(proj=proj)
    if pred_objpoints3d is not None:
        ax.scatter(
            pred_objpoints3d[:, proj_1],
            pred_objpoints3d[:, proj_2],
            c="r",
            alpha=0.1,
            s=1,
        )
    # ax.set_aspect("equal")


def get_proj_axis(proj="z"):
    if proj == "z":
        proj_1 = 0
        proj_2 = 1
    elif proj == "y":
        proj_1 = 0
        proj_2 = 2
    elif proj == "x":
        proj_1 = 1
        proj_2 = 2
    return proj_1, proj_2


def visualize_joints_2d(
    ax, joints, joint_idxs=True, links=None, alpha=1, scatter=True, linewidth=2
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth)
    ax.axis("equal")


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    colors = ["r", "m", "b", "c", "g"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )
