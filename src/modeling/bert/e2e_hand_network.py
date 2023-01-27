"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


class Graphormer_Hand_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for hand pose and mesh reconstruction from a single image.
    '''

    def __init__(self, config, backbone, trans_encoder):
        super(Graphormer_Hand_Network, self).__init__()
        self.config = config
        self.backbone = backbone
        self.cam_param_fc = torch.nn.Linear(3, 1)    
        self.cam_param_fc2 = torch.nn.Linear(21, 3)
        self.trans_encoder = trans_encoder
        self.grid_feat_dim = torch.nn.Linear(1024, 2048)

    def forward(self, images):
        batch_size = images.size(0)

        image_feat, grid_feat = self.backbone(images)
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, 21, -1)

        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)
        grid_feat = self.grid_feat_dim(grid_feat)

        features = torch.cat([image_feat, grid_feat], dim=1)
        features = self.trans_encoder(features)

        pred_3d_joints = features[:, :21, :]
        x = self.cam_param_fc(pred_3d_joints)
        x = x.transpose(1, 2)
        x = self.cam_param_fc2(x)
        cam_param = x.transpose(1, 2)
        cam_param = cam_param.squeeze()
        pred_2d_joints = orthographic_projection(pred_3d_joints, cam_param)
        
        return pred_2d_joints, pred_3d_joints
