import gc
import os
import sys

import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from torch.utils import data
from src.utils.dir import reset_folder
from torch.utils.tensorboard import SummaryWriter
from argparser import parse_args, load_model, train, valid, pred_store, pred_eval
from dataset import *

def main(args, logger):


    train_dataset, test_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    if not args.eval:
        _model, best_loss, epo, count = load_model(args)
        log_dir = f'tensorboard/{args.output_dir}'; reset_folder(log_dir)
        writer = SummaryWriter(log_dir); pck_l = 0; batch_time = AverageMeter()

        for epoch in range(epo, args.epoch):
            Graphormer_model, optimizer, batch_time = train(args, trainset_loader, _model, epoch, best_loss, len(train_dataset),logger, count, writer, pck_l, len(trainset_loader)+len(testset_loader), batch_time)
            loss, count, pck, batch_time = valid(args, testset_loader, Graphormer_model, epoch, count, best_loss, len(train_dataset), logger, writer, batch_time, len(trainset_loader)+len(testset_loader), pck_l)
            
            pck_l = max(pck, pck_l)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            if is_best:
                count = 0
                _model = Graphormer_model
                save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, count,  'good',logger=logger)
                del Graphormer_model

            else:
                count += 1
                if count == args.count:
                    break

            gc.collect()
            torch.cuda.empty_cache()
    else:
        model_path = "final_model"
        model_list, eval_list = list(), list()
        for (root, _, files) in os.walk(model_path):
            for file in files:
                if '.bin' in file:
                    model_list.append(os.path.join(root, file))
                    
        pbar = tqdm.tqdm(total = len(model_list)) 
        for path_name in model_list:
            args.model = path_name.split('/')[1]
            args.name = ('/').join(path_name.split('/')[1:-2])
            args.output_dir = os.path.join(model_path, args.name)
            _model, _, _, _ = load_model(args)
            state_dict = torch.load(path_name)
            _model.load_state_dict(state_dict['model_state_dict'], strict=False)
            pred_store(args, testset_loader, _model)        
            T_list = np.arange(0.05, 0.25, 0.05, dtype=np.float32)
            threshold_type = 'proportion'
            pck_list, epe_list = pred_eval(args, T_list, threshold_type)
            eval_list.append([args.name, pck_list, epe_list])
            pbar.update(1)
            
        pbar.close()
            
        pbar.close()
        f = open(f"pck_{threshold_type}.txt", "w")
        for each_list in eval_list:
            epe_list, pck_list, list_name = each_list[2], each_list[1], each_list[0]
            for p_type in pck_list:
                T = [key for key in pck_list['%s'%p_type].keys()]
                f.write("{}; {};".format(p_type, list_name))
                for j in range(len(T_list)): f.write("{:.2f},".format(pck_list['%s'%p_type][T[j]]))
                f.write("{:.2f}, {:.2f}\n".format(epe_list['%s'%p_type], pck_list['%s'%p_type][T[0]]))
        f.close()

if __name__ == "__main__":
    args, logger = parse_args()
    main(args, logger)
