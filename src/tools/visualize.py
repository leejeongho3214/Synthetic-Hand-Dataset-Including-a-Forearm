import cv2
import numpy as np
from matplotlib import pyplot as plt


def visualize_prediction(images, pred_2d_joint, fig):
    image = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[0][i][0]), int(pred_2d_joint[0][i][1])),
                     (int(pred_2d_joint[0][parents[i]][0]), int(pred_2d_joint[0][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('pred_image')
    ax1.axis("off")
    plt.show()

def visualize_gt(images, gt_2d_joint, fig):
    image = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,])
    for i in range(21):
        cv2.circle(image, (int(gt_2d_joint[0][i][0]), int(gt_2d_joint[0][i][1])), 2, [0, 1, 0],
                   thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[0][i][0]), int(gt_2d_joint[0][i][1])),
                     (int(gt_2d_joint[0][parents[i]][0]), int(gt_2d_joint[0][parents[i]][1])),
                     [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image[:, :, (2, 1, 0)])
    ax1.set_title('gt_image')
    ax1.axis("off")
