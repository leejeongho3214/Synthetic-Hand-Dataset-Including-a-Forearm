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
os.environ["CUDA_VISIBLE_DEVICES"]= "3" 

def main(args):
    count = 0
    _model, logger, best_loss, epo = load_model(args)

    train_dataloader, test_dataloader, train_dataset1, test_dataset1 = make_hand_data_loader(args, args.train_yaml,
                                                                          args.distributed, is_train=True,
                                                                          scale_factor=args.img_scale_factor) ## RGB image
    path = "../../datasets/1023/org"
    folder_num = os.listdir(path)

    for iter, degree in enumerate(folder_num):

        dataset = Json_transform(degree, path, rotation = True, color= True, color_num = 1, background = True)
        # if iter == 0:
        #     train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

        # else:
        #     train_dataset_new, test_dataset_new = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
        #     train_dataset  = ConcatDataset([train_dataset, train_dataset_new])        
        #     test_dataset = ConcatDataset([test_dataset, test_dataset_new])    
        
    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
    # test_dataset = CustomDataset_test()
    ## trainset = CISLAB_HAND , train_dataset = FreiHAND
    # concat_dataset = ConcatDataset([trainset, train_dataset])

    # dataset = HIU_Dataset_align()
    # train_dataset1, test_dataset1 = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-(int(len(dataset)*0.9))])
    assert False, "finish"
    train_dataset = ConcatDataset([train_dataset1, train_dataset])
    test_dataset = ConcatDataset([test_dataset1, test_dataset])
    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    logger.info("Name: {} // loss_2d: {} // loss_3d: {} // Train_length: {} // Test_length: {} \n".format(args.name, args.loss_2d, args.loss_3d, len(train_dataset), len(test_dataset)))

    for epoch in range(epo, 1000):
        Graphormer_model, optimizer = train(args, trainset_loader, _model, epoch, best_loss, len(train_dataset),logger, count)
        loss, count = test(args, testset_loader, Graphormer_model, epoch, count, best_loss,logger)
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