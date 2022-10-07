"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg

class Graphormer_Hand_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for hand pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, token, camera= False):
        super(Graphormer_Hand_Network, self).__init__()
        self.config = config
        self.token = token
        self.camera = camera
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(token, 3)
        # self.cam_param_fc3 = torch.nn.Linear(3, 1)
        # self.cam_param_fc4 = torch.nn.Linear(21, 1)
        # self.cam_param_fc5 = torch.nn.Linear(3, 1)
        # self.cam_param_fc6= torch.nn.Linear(21, 3)
        # self.cam_param_fc3 = torch.nn.Linear(150, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2048)

    def forward(self, images, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh

        # import matplotlib.pyplot as plt
        # import numpy as np
        # a = images[3].cpu().numpy()
        # a = np.moveaxis(a, 0, -1)
        # plt.imshow(a)
        # plt.show()

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # concatinate image feat and mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, 21, -1)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        # features = image_feat
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([image_feat, grid_feat],dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s  
            special_token = torch.ones_like(features[:,:-49,:]).cuda()*0.01
            features[:,:-49,:] = features[:,:-49,:]*meta_masks + special_token*(1-meta_masks)

        # forward pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:21,:]
        # pred_vertices_sub = features[:,num_joints:-49,:]

        if self.token == 49:
        # learn camera parameters
            x = self.cam_param_fc(features[:,21:,:])
            x = x.transpose(1,2)
            x = self.cam_param_fc2(x)
        else:
            x = self.cam_param_fc(features[:, :, :])
            x = x.transpose(1, 2)
            x = self.cam_param_fc2(x)

        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze()

           # return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
        return cam_param, pred_3d_joints