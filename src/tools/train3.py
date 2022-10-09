import gc
from torch.utils import data
from torch.utils.data import random_split, ConcatDataset
import sys
from argparser import parse_args, load_model, train, test
from dataset import *
from src.datasets.build import make_hand_data_loader
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
# sys.path.append("C:\\Users\\jeongho\\PycharmProjects\\PoseEstimation\\HandPose\\MeshGraphormer-main")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use



def main(args):
    count = 0
    _model, logger, best_loss, epo = load_model(args)

    # train_dataloader, test_dataloader, train_dataset, test_dataset = make_hand_data_loader(args, args.train_yaml,
    #                                                                       args.distributed, is_train=True,
    #                                                                       scale_factor=args.img_scale_factor)

    folder_num = os.listdir("../../datasets/CISLAB_various")

    for iter, degree in enumerate(folder_num):
        
        dataset = CustomDataset_train_new(degree)
        if iter == 0:
            train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        else:
            train_dataset_new, test_dataset_new = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
            train_dataset  = ConcatDataset([train_dataset, train_dataset_new])        
            test_dataset = ConcatDataset([test_dataset, test_dataset_new])    
        
    train_dataloader, test_dataloader, train_dataset_frei, test_dataset_frei = make_hand_data_loader(args, args.train_yaml,
                                                                        args.distributed, is_train=True,
                                                                        scale_factor=args.img_scale_factor)
    # dataset = CustomDataset_train()
    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
    # test_dataset = CustomDataset_test()
    ## trainset = CISLAB_HAND , train_dataset = FreiHAND
    dataset = HIU_Dataset_Align()
    train_dataset_hiu, test_dataset_hiu = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-(int(len(dataset)*0.9))])
    train_both = ConcatDataset([train_dataset_hiu, train_dataset])
    test_both = ConcatDataset([test_dataset_hiu, test_dataset])

    trainset_loader = data.DataLoader(dataset=train_both, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_both, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)



    # train_dataset, test_dataset = random_split(dataset_new, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
    # test_dataset = CustomDataset_test()
    ## trainset = CISLAB_HAND , train_dataset = FreiHAND
    # concat_dataset = ConcatDataset([trainset, train_dataset])

    # dataset = HIU_Dataset()
    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-(int(len(dataset)*0.9))])


    for epoch in range(epo, 1000):
        Graphormer_model, optimizer = train(args, trainset_loader, _model, epoch, best_loss, len(train_dataset),logger)
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
