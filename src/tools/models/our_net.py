import torch
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.our_hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat as get_cls_net_gridfeat_our
from src.modeling.our_hrnet import update_config_our as hrnet_update_config
from src.modeling.our_hrnet import config as hrnet_config
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
import torchvision.models as models
from src.tools.models.simplebaseline import get_pose_net

def get_our_net(args):
    trans_encoder = []
    input_feat_dim = [int(item) for item in args.input_feat_dim]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim]
    output_feat_dim = input_feat_dim[1:] + [3] ## origin => change to input_feat_dim

    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
    # init three transformer-encoder blocks in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, Graphormer
        config = config_class.from_pretrained(args.config_name if args.config_name \
                                                else args.model_name_or_path)

        config.output_attentions = True
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size * 2)

        if which_blk_graph[i] == 1:
            config.graph_conv = True
            # logger.info("Add Graph Conv")
        else:
            config.graph_conv = False


        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                # logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)



        assert args.graph_num <= config.num_hidden_layers, "graph_num is more than hidden_layers"
        config.graph_num = args.graph_num
        config.mesh_type = args.mesh_type


        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)
        trans_encoder.append(model)

        if args.backbone == 'hrnet':
            # create backbone model
            if args.arch == 'hrnet':
                hrnet_yaml = '../../models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                hrnet_checkpoint = '../../models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
                hrnet_update_config(hrnet_config, hrnet_yaml)
                backbone = get_cls_net_gridfeat_our(hrnet_config, pretrained=hrnet_checkpoint)
                # logger.info('=> loading hrnet-v2-w40 model')
            elif args.arch == 'hrnet-w64':
                hrnet_yaml = '../../models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                hrnet_checkpoint = '../../models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
                hrnet_update_config(hrnet_config, hrnet_yaml)
                backbone = get_cls_net_gridfeat_our(hrnet_config, pretrained=hrnet_checkpoint)
                # logger.info('=> loading hrnet-v2-w64 model')
            else:
                print("=> using pre-trained model '{}'".format(args.arch))
                backbone = models.__dict__[args.arch](pretrained=True)
                # remove the last fc layer
                backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        else:
            from core.config import config
            from core.config import update_config
            
            update_config("models/224x224.yaml")
            backbone = get_pose_net(config, is_train = True)

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    _model = Graphormer_Network(config, backbone, trans_encoder)
    
    return _model
