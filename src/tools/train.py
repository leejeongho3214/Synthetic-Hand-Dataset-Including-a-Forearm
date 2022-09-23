import gc
from torch.utils import data
from torch.utils.data import random_split
import sys
from argparser import parse_args, load_model, train, test
from dataset import *

sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
# sys.path.append("C:\\Users\\jeongho\\PycharmProjects\\PoseEstimation\\HandPose\\MeshGraphormer-main")



def main(args):
    count = 0
    _model, logger, best_loss, epo = load_model(args)

    # train_dataloader, test_dataloader, train_dataset, test_dataset = make_hand_data_loader(args, args.train_yaml,
    #                                                                       args.distributed, is_train=True,
    #                                                                       scale_factor=args.img_scale_factor)
    # trainset = CustomDataset_train(args)
    # testset = CustomDataset_test()
    ## trainset = CISLAB_HAND , train_dataset = FreiHAND
    # concat_dataset = ConcatDataset([trainset, train_dataset])

    dataset = HIU_Dataset()
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-(int(len(dataset)*0.9))])
    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    for epoch in range(epo, 1000):
        Graphormer_model, optimizer, _ = train(args, trainset_loader, _model, epoch, best_loss, len(train_dataset),logger)
        loss = test(args, testset_loader, Graphormer_model, epoch, count, best_loss,logger)
        is_best = loss > best_loss
        best_loss = max(loss, best_loss)
        if is_best:
            count = 0
            _model = Graphormer_model
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, 'good',logger=logger)
            del Graphormer_model

        else:
            count += 1
            if count == 50:
                break

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
