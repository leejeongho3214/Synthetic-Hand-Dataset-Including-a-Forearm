import torch
from src.utils.loss import adjust_learning_rate
import numpy as np
import time 
from time import ctime
from src.utils.visualize import *
from src.utils.metric_logger import AverageMeter
from src.utils.loss import *
from src.utils.bar import *

class Runner(object):
    def __init__(self, args, model, epoch, data_loader, phase, batch_time):
        super(Runner, self).__init__()
        self.args = args
        self.data = data_loader
        self.batch_time = batch_time
        self.bar = Bar(colored(str(epoch)+'_EPOCH_'+phase, color='blue'), max=len(data_loader))
        self.model = model
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)
        self.epoch = epoch
        self.criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
        self.log_losses = AverageMeter()
        self.log_2d_losses = AverageMeter()
        self.log_3d_losses = AverageMeter()
        self.log_3d_re_losses = AverageMeter()
        self.pck_losses = AverageMeter()
        self.epe_losses = AverageMeter()
        
    def train_log(self, dataloader, logger, data_len, iteration, count, pck , best_loss, eta_seconds, end, epoch):
        if iteration % self.args.logging_steps == 0:
            logger.debug( 
                         ' '.join(
                            ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi}, count: {count}/{max_count}']
                        ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(dataloader), count= count, max_count = self.args.count)
                        + ' 2d_loss: {:.8f}, 3d_loss: {:.8f}, 3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                            self.log_2d_losses.avg,
                            self.log_3d_losses.avg,
                            self.log_3d_re_losses.avg,
                            pck,
                            self.log_losses.avg,
                            best_loss,
                            ctime(eta_seconds + end)))
        
        self.bar.suffix = ('({iteration}/{data_loader}) '
                           'name: {name} | '
                           'type: {d_type} | '
                           'count: {count} | '
                           'loss: {total:.8f} | '
                           'best_pck: {pck:.2f} | '
                           'exp: {exp}'
                           ).format(name= self.args.name.split('/')[-1], count = count, max_count = self.args.count, iteration = iteration, exp = ctime(eta_seconds + end),
                                    data_loader = len(self.data), total = self.log_losses.avg, pck = pck, d_type = "3D" if self.args.projection else "2D")

        self.bar.next()
        
    def test_log(self, dataloader, logger, iteration, count, best_loss, eta_seconds, end, epoch, pck):
        if iteration % (self.args.logging_steps /10) == 0:
            logger.debug(
                        ' '.join(
                            ['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                        ).format(ep=epoch, iter=iteration, maxi=len(dataloader))
                        + ' pck: {:.2f}%, epe: {:.2f}mm, count: {} / {}, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                            self.pck_losses.avg * 100,
                            self.epe_losses.avg * 0.26,
                            int(count),
                            self.args.count,
                            self.log_losses.avg,
                            best_loss,
                            ctime(eta_seconds + end))
                        )

        self.bar.suffix = ('({iteration}/{data_loader}) '
                           'name: {name} | '
                           'type: {d_type} | '
                           'count: {count} | '
                           'loss: {total:.8f} | '
                           'pck: {now_pck:.2f} | '
                           'best_pck: {pck:.2f} | '
                           'exp: {exp}'
                           ).format(name= self.args.name.split('/')[-1], count = count, max_count = self.args.count, iteration = iteration, exp = ctime(eta_seconds + end),
                                    data_loader = len(self.data), total = self.log_losses.avg,now_pck = self.pck_losses.avg * 100 ,pck = pck * 100, d_type = "3D" if self.args.projection else "2D")
        if iteration == len(dataloader) - 1:
            self.bar.suffix = self.bar.suffix +'\n'
        self.bar.next()
    
    def our(self, dataloader, end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase = 'train'):
        if phase == 'TRAIN':
            self.model.train()
            for iteration, (images, gt_2d_joints, _, gt_3d_joints) in enumerate(dataloader):
                batch_size = images.size(0)
                adjust_learning_rate(self.optimizer, self.epoch, self.args)  
                gt_2d_joints[:,:,1] = gt_2d_joints[:,:,1] / images.size(2) ## You Have to check whether weight and height is correct dimenstion
                gt_2d_joints[:,:,0] = gt_2d_joints[:,:,0] / images.size(3)  ## 2d joint value rearrange from 0 to 1
                gt_2d_joint = gt_2d_joints.cuda()
                gt_3d_joints = gt_3d_joints.cuda()
                
                parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
                gt_3d_mid_joints = torch.ones(batch_size, 20, 3)
                for i in range(20):
                    gt_3d_mid_joints[:, i, :] =  (gt_3d_joints[:, i + 1, :] + gt_3d_joints[:, parents[i + 1], :]) / 2
                
                if self.args.scale:
                    scale = ((gt_3d_joints[:, 10,:] - gt_3d_joints[:, 9, :])**2).sum(-1).sqrt()
                    for i in range(batch_size):
                        gt_3d_joints[i] = gt_3d_joints[i]/scale[i]

                images = images.cuda()
                
                if self.args.projection: pred_2d_joints, pred_3d_joints= self.model(images)
                else: pred_2d_joints= self.model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); self.args.loss_3d = 0

                pred_3d_mid_joints = torch.ones(batch_size, 20, 3)
                for i in range(20):
                    pred_3d_mid_joints[:, i, :] =  (pred_3d_joints[:, i + 1, :] + pred_3d_joints[:, parents[i + 1], :]) / 2
                
                loss_2d = keypoint_2d_loss(self.criterion_keypoints, pred_2d_joints, gt_2d_joint)
                loss_3d = keypoint_3d_loss(self.criterion_keypoints, pred_3d_mid_joints, gt_3d_mid_joints)
                loss_3d_re = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
                loss_3d_mid = keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints, gt_3d_joints)
                
                if self.args.projection:
                    loss = self.args.loss_2d * loss_2d + self.args.loss_3d * loss_3d + self.args.loss_3d_mid * loss_3d_mid
                else:
                    loss = loss_2d
                    
                self.log_losses.update(loss.item(), batch_size)
                self.log_2d_losses.update(loss_2d.item(), batch_size)
                self.log_3d_losses.update(loss_3d.item(), batch_size)
                self.log_3d_re_losses.update(loss_3d_re.item(), batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)
                gt_2d_joint[:,:,1] = gt_2d_joint[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                gt_2d_joint[:,:,0] = gt_2d_joint[:,:,0] * images.size(3) 
                
                if not self.args.projection:
                    if iteration == 0 or iteration == int(len(dataloader)/2) or iteration == len(dataloader) - 1:
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joint, fig, iteration)
                        visualize_pred(images, pred_2d_joints, fig, 'train', self.epoch, iteration, self.args, None)
                        plt.close()

                self.batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = self.batch_time.avg * ((len_total - iteration) + (self.args.epoch - epoch -1) * len_total)  

                self.train_log(dataloader, logger, data_len, iteration, count, pck , best_loss, eta_seconds, end, epoch)
                if iteration == len(dataloader) - 1:
                    writer.add_scalar("Loss/train", self.log_losses.avg, epoch)

            return self.model, self.optimizer, self.batch_time
            
        else:
            self.model.eval()
            with torch.no_grad():
                for iteration, (images, gt_2d_joints, _, gt_3d_joints) in enumerate(dataloader):
                    batch_size = images.size(0)
                    
                    images = images.cuda()
                    gt_2d_joint = gt_2d_joints.cuda()
                    gt_3d_joints = torch.tensor(gt_3d_joints).cuda()

                    if self.args.projection: 
                        pred_2d_joints, pred_3d_joints= self.model(images)
                        pck, _ = PCK_3d_loss(pred_3d_joints, gt_3d_joints, T= 10)
                        loss = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
                        
                    else: 
                        pred_2d_joints= self.model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); self.args.loss_3d = 0
                        pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                        pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)
                        pck = PCK_2d_loss_visible(pred_2d_joints, gt_2d_joint, T= 0.05, threshold = 'proportion')
                        loss = keypoint_2d_loss(self.criterion_keypoints, pred_2d_joints, gt_2d_joint)
                        
                    epe_loss, _ = EPE_train(pred_2d_joints, gt_2d_joint)  ## consider invisible joint

                    self.pck_losses.update(pck, batch_size)
                    self.epe_losses.update_p(epe_loss[0], epe_loss[1])
                    self.log_losses.update(loss.item(), batch_size)
                    
                    if not self.args.projection:
                        if iteration == 0 or iteration == int(len(dataloader)/2) or iteration == len(dataloader) - 1:
                            fig = plt.figure()
                            visualize_gt(images, gt_2d_joint, fig, iteration)
                            visualize_pred(images, pred_2d_joints, fig, 'test', epoch, iteration, self.args,None)
                            plt.close()
                            
                    self.batch_time.update(time.time() - end)
                    end = time.time()
                    eta_seconds = self.batch_time.avg * ((len(dataloader) - iteration) + (self.args.epoch - epoch -1) *len_total)

                    self.test_log(dataloader, logger, iteration, count, best_loss, eta_seconds, end, epoch, pck)
                    if iteration == len(dataloader) - 1:
                        writer.add_scalar("Loss/valid", self.log_losses.avg, epoch)

                return self.log_losses.avg, count, self.pck_losses.avg * 100, self.batch_time

            
                
    def other(self, dataloader, end, epoch, logger, data_len, len_total, count, pck, best_loss, writer, phase = 'train'):
        heatmap_size, multiply = 64, 4
        if self.args.model == "hrnet": heatmap_size, multiply = 128, 2
        if phase == "TRAIN":
            self.model.train()
            for iteration, (images, gt_2d_joints, gt_heatmaps, _) in enumerate(dataloader):
                batch_size = images.size(0)
                adjust_learning_rate(self.optimizer, epoch, self.args)
                images = images.cuda()
                gt_heatmaps = gt_heatmaps.cuda()     
                pred = self.model(images)

                loss = torch.mean(calc_loss(pred, gt_heatmaps, self.args))

                self.log_losses.update(loss.item(), batch_size)

                if self.args.model == "hourglass": pred = pred[:, -1]

                pred_joint = np.zeros((pred.size(0),pred.size(1),2))
                for idx, batch in enumerate(pred):
                    for idx2, joint in enumerate(batch):
                        joint = joint.detach().cpu()
                        joint = joint.flatten()
                        index = joint.argmax()
                        row = int(index / heatmap_size)
                        col = index % heatmap_size
                        pred_joint[idx,idx2] = np.array([col, row]).flatten()
                pred_joint = torch.tensor(pred_joint)
                pred_joint = pred_joint * multiply ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
                
                # back prop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.args.projection:
                    if iteration == 0 or iteration == int(len(dataloader)/2) or iteration == len(dataloader) - 1:
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joints, fig, iteration)
                        visualize_pred(images, pred_joint, fig, 'train', epoch, iteration, self.args,None)
                        plt.close()

                self.batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = self.batch_time.avg * ((len_total - iteration) + (self.args.epoch - epoch -1) * len_total)  

                self.train_log(dataloader, logger, data_len, iteration, count, pck , best_loss, eta_seconds, end, epoch)
                if iteration == len(dataloader) - 1:
                    writer.add_scalar("Loss/train", self.log_losses.avg, epoch)
            return self.model, self.optimizer, self.batch_time

        else:
            self.model.eval()
            for iteration, (images, gt_2d_joints, gt_heatmaps, _) in enumerate(dataloader):
                batch_size = images.size(0)
                images = images.cuda()
                gt_2d_joint = gt_2d_joints.cuda()
                gt_heatmaps = gt_heatmaps.cuda()    
                
                pred = self.model(images)
                
                loss = torch.mean(calc_loss(pred, gt_heatmaps, self.args))
                self.log_losses.update(loss.item(), batch_size)
                if self.args.model == "hourglass": pred = pred[:, -1]

                pred_joint = np.zeros((pred.size(0),pred.size(1),2))
                for idx, batch in enumerate(pred):
                    for idx2, joint in enumerate(batch):
                        joint = joint.detach().cpu()
                        joint = joint.flatten()
                        index = joint.argmax()
                        row = int(index / heatmap_size)
                        col = index % heatmap_size
                        pred_joint[idx,idx2] = np.array([col, row]).flatten()
                pred_joint = torch.tensor(pred_joint)
                pred_2d_joints = pred_joint * multiply ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
                
                pck = PCK_2d_loss(pred_2d_joints, gt_2d_joint, T= 0.05, threshold='proportion')
                epe_loss, _ = EPE_train(pred_2d_joints, gt_2d_joint)  ## consider invisible joint
                self.pck_losses.update(pck, batch_size)
                self.epe_losses.update_p(epe_loss[0], epe_loss[1])
                
                if not self.args.projection:
                    if iteration == 0 or iteration == int(len(dataloader)/2) or iteration == len(dataloader) - 1:
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joints, fig, iteration)
                        visualize_pred(images, pred_2d_joints, fig, 'test', epoch, iteration, self.args, None)
                        plt.close()

                self.batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = self.batch_time.avg * ((len_total - iteration) + (self.args.epoch - epoch -1) * len_total)  

                self.test_log(dataloader, logger, iteration, count, best_loss, eta_seconds, end, epoch, pck)
                if iteration == len(dataloader) - 1:
                    writer.add_scalar("Loss/valid", self.log_losses.avg, epoch)
                    
            return self.log_losses.avg, count, self.pck_losses.avg * 100, self.batch_time 
