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
    def __init__(self, args, model, epoch, train_loader, valid_loader,  phase, batch_time,  data_len, len_total, count, pck, best_loss, writer):
        super(Runner, self).__init__()
        self.args = args
        self.len_data = data_len
        self.len_total = len_total
        self.count = count
        self.pck = pck
        self.best_loss = best_loss
        self.phase = phase
        self.writer = writer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.batch_time = batch_time
        self.now_loader = train_loader if phase == "TRAIN" else valid_loader
        self.bar = Bar(colored(str(epoch) + '_' + phase,
                       color='blue'), max=len(self.now_loader))
        self.model = model
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=args.lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=0)
        self.epoch = epoch
        self.criterion_keypoints = torch.nn.MSELoss(
            reduction='none').cuda(args.device)
        self.log_losses = AverageMeter()
        self.log_2d_losses = AverageMeter()
        self.log_3d_losses = AverageMeter()
        self.log_3d_re_losses = AverageMeter()
        self.pck_losses = AverageMeter()
        self.epe_losses = AverageMeter()

    def train_log(self, iteration, eta_seconds, end):
        tt = ' '.join(ctime(eta_seconds + end).split(' ')[1:-1])

        if iteration % (self.args.logging_steps * 5) == 0:
            self.args.logger.debug(
                ' '.join(
                    ['dataset_length: {len}', 'epoch: {ep}',
                        'iter: {iter}', '/{maxi}, count: {count}/{max_count}']
                ).format(len=self.len_data, ep=self.epoch, iter=iteration, maxi=len(self.now_loader), count=self.count, max_count=self.args.count)
                + ' 2d_loss: {:.8f}, 3d_loss: {:.8f}, 3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}'.format(
                    self.log_2d_losses.avg,
                    self.log_3d_losses.avg,
                    self.log_3d_re_losses.avg,
                    self.pck,
                    self.log_losses.avg,
                    self.best_loss))

        if iteration == 0:
            self.bar.suffix = ('({iteration}/{data_loader}) '
                               'name: {name} | '
                               'count: {count} | '
                               'loss: {total:.6f} \n'
                               ).format(name="/".join(self.args.name.split('/')[-2:]), count=self.count, iteration=iteration, exp=tt,
                                        data_loader=len(self.now_loader), total=self.log_losses.avg)
        else:
            self.bar.suffix = ('({iteration}/{data_loader}) '
                               'loss: {total:.6f} '
                               ).format(name="/".join(self.args.name.split('/')[-2:]), count=self.count, iteration=iteration, exp=tt,
                                        data_loader=len(self.now_loader), total=self.log_losses.avg)
        self.bar.next()

    def test_log(self, iteration, eta_seconds, end):
        tt = ' '.join(ctime(eta_seconds + end).split(' ')[1:-1])
        if iteration % (self.args.logging_steps / 2) == 0:
            self.args.logger.debug(
                ' '.join(['Test =>> epoch: {ep}', 'iter: {iter}', '/{maxi}']
                         ).format(ep=self.epoch, iter=iteration, maxi=len(self.now_loader))
                + ' epe: {:.2f}mm, count: {} / {}, total_loss: {:.8f}, best_loss: {:.8f}'.format(
                    self.epe_losses.avg * 0.26,
                    int(self.count),
                    self.args.count,
                    self.log_losses.avg,
                    self.best_loss)
            )

        if iteration == 0:
            self.bar.suffix = ('({iteration}/{data_loader}) '
                               'name: {name} | '
                               'count: {count} | '
                               'best_loss: {best_loss:.6f} \n'
                               ).format(name="/".join(self.args.name.split('/')[-2:]), count=self.count,  iteration=iteration, best_loss=self.best_loss,
                                        data_loader=len(self.now_loader), total=self.log_losses.avg)
        else:
            self.bar.suffix = ('({iteration}/{data_loader}) '
                               'loss: {total:.6f} '
                               ).format(name="/".join(self.args.name.split('/')[-2:]), count=self.count,  iteration=iteration, best_loss=self.best_loss,
                                        data_loader=len(self.now_loader), total=self.log_losses.avg)
        self.bar.next()

    def our(self, end):
        if self.phase == 'TRAIN':
            self.model.train()
            for iteration, (images, gt_2d_joints, gt_3d_joints) in enumerate(self.train_loader):
                batch_size = images.size(0)
                adjust_learning_rate(self.optimizer, self.epoch, self.args)

                gt_2d_joint = (gt_2d_joints).cuda()
                gt_3d_joints = gt_3d_joints.cuda()
                images = images.cuda()
                

                pred_2d_joints, pred_3d_joints = self.model(images)

                loss_2d = keypoint_2d_loss(
                    self.criterion_keypoints, pred_2d_joints, gt_2d_joint)
                loss_3d = keypoint_3d_loss(
                    self.criterion_keypoints, pred_3d_joints, gt_3d_joints)
                loss = loss_3d * self.args.loss_3d + loss_2d * self.args.loss_2d

                self.log_losses.update(loss.item(), batch_size)
                self.log_2d_losses.update(loss_2d.item(), batch_size)
                self.log_3d_losses.update(loss_3d.item(), batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                gt_2d_joint = gt_2d_joint * 224
                if iteration == 0 or iteration == int(len(self.train_loader)/2) or iteration == len(self.train_loader) - 1:
                    fig = plt.figure()
                    visualize_only_gt(images, gt_2d_joint, fig,
                                      'train', self.epoch, iteration, self.args)

                    plt.close()

                self.batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = self.batch_time.avg * \
                    ((self.len_total - iteration) +
                     (self.args.epoch - self.epoch - 1) * self.len_total)

                self.train_log(iteration, eta_seconds, end)
            self.writer.add_scalar(
                f"Loss/train", self.log_losses.avg, self.epoch)

            return self.model, self.optimizer, self.batch_time

        else:
            self.model.eval()
            with torch.no_grad():
                for iteration, (images, gt_2d_joints, gt_3d_joints) in enumerate(self.valid_loader):
                    batch_size = images.size(0)

                    images = images.cuda()
                    gt_2d_joint = gt_2d_joints.cuda()
                    gt_3d_joints = gt_3d_joints.cuda()

                    pred_2d_joints, pred_3d_joints = self.model(images)
                    loss_2d = keypoint_2d_loss(
                        self.criterion_keypoints, pred_2d_joints, gt_2d_joint)
                    loss_3d = keypoint_3d_loss(
                        self.criterion_keypoints, pred_3d_joints, gt_3d_joints)
                    loss = loss_3d * self.args.loss_3d + loss_2d * self.args.loss_2d

                    gt_2d_joint = gt_2d_joint * 224
                    if iteration == 0 or iteration == int(len(self.valid_loader)/2) or iteration == len(self.valid_loader) - 1:
                        fig = plt.figure()
                        visualize_only_gt(
                            images, gt_2d_joint, fig, 'test', self.epoch, iteration, self.args)
                        plt.close()

                    self.log_losses.update(loss.item(), batch_size)
                    self.batch_time.update(time.time() - end)

                    end = time.time()
                    eta_seconds = self.batch_time.avg * \
                        ((len(self.valid_loader) - iteration) +
                         (self.args.epoch - self.epoch - 1) * self.len_total)

                    self.test_log(iteration, eta_seconds, end)

                self.writer.add_scalar(
                    "Loss/valid", self.log_losses.avg, self.epoch)

                return self.log_losses.avg, self.count, self.pck_losses.avg * 100, self.batch_time
