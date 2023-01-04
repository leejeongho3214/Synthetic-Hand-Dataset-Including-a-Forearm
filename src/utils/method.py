import torch
from src.utils.loss import adjust_learning_rate

class Runner(object):
    def init(self,args, model, optimizer, train_dataloader, epoch, criterion_2d_keypoints, criterion_keypoints, log_2d_losses, log_3d_losses, log_3d_re_losses, end):
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.epoch = epoch
        self.criterion_2d_keypoints = criterion_2d_keypoints
        self.criterion_keypoints = criterion_keypoints
        self.log_2d_losses = log_2d_losses
        self.log_3d_losses = log_3d_losses
        self.log_3d_re_losses = log_3d_re_losses
        self.end = end
        
    def our_train(self):
        for iteration, (images, gt_2d_joints, _, gt_3d_joints) in enumerate(self.train_dataloader):
                batch_size = images.size(0)
                adjust_learning_rate(self.optimizer, self.epoch, self.self.args)  
                gt_2d_joints[:,:,1] = gt_2d_joints[:,:,1] / images.size(2) ## You Have to check whether weight and height is correct dimenstion
                gt_2d_joints[:,:,0] = gt_2d_joints[:,:,0] / images.size(3)  ## 2d joint value rearrange from 0 to 1
                gt_2d_joint = gt_2d_joints.clone().detach()
                gt_2d_joint = gt_2d_joint.cuda()
                gt_3d_joints = gt_3d_joints.clone().detach()
                gt_3d_joints = gt_3d_joints.cuda()
                
                parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
                gt_3d_mid_joints = torch.ones(batch_size, 20, 3)
                for i in range(20):
                    gt_3d_mid_joints[:, i, :] =  (gt_3d_joints[:, i + 1, :] + gt_3d_joints[:, parents[i + 1], :]) / 2
                
                if self.self.args.scale:
                    scale = ((gt_3d_joints[:, 10,:] - gt_3d_joints[:, 9, :])**2).sum(-1).sqrt()
                    for i in range(batch_size):
                        gt_3d_joints[i] = gt_3d_joints[i]/scale[i]

                images = images.cuda()
                
                if self.self.args.projection: pred_2d_joints, pred_3d_joints= self.model(images)
                else: pred_2d_joints= self.model(images); pred_3d_joints = torch.zeros([pred_2d_joints.size()[0], pred_2d_joints.size()[1], 3]).cuda(); self.args.loss_3d = 0

                pred_3d_mid_joints = torch.ones(batch_size, 20, 3)
                for i in range(20):
                    pred_3d_mid_joints[:, i, :] =  (pred_3d_joints[:, i + 1, :] + pred_3d_joints[:, parents[i + 1], :]) / 2
                
                loss_2d= keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joint)
                loss_3d = keypoint_3d_loss(criterion_keypoints, pred_3d_mid_joints, gt_3d_mid_joints)
                loss_3d_re = reconstruction_error(np.array(pred_3d_joints.detach().cpu()), np.array(gt_3d_joints.detach().cpu()))
                loss_3d_mid = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
                if self.args.projection:
                    loss = self.args.loss_2d * loss_2d + self.args.loss_3d * loss_3d + self.args.loss_3d_mid * loss_3d_mid
                else:
                    loss = loss_2d
                log_losses.update(loss.item(), batch_size)
                log_2d_losses.update(loss_2d.item(), batch_size)
                log_3d_losses.update(loss_3d.item(), batch_size)
                log_3d_re_losses.update(loss_3d_re.item(), batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_2d_joints[:,:,1] = pred_2d_joints[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                pred_2d_joints[:,:,0] = pred_2d_joints[:,:,0] * images.size(3)
                gt_2d_joint[:,:,1] = gt_2d_joint[:,:,1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
                gt_2d_joint[:,:,0] = gt_2d_joint[:,:,0] * images.size(3) 
                
                if not self.args.projection:
                    if iteration == 0 or iteration == int(len(train_dataloader)/2) or iteration == len(train_dataloader) - 1:
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joint, fig, iteration)
                        visualize_pred(images, pred_2d_joints, fig, 'train', epoch, iteration, self.args, None)
                        plt.close()

                batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = batch_time.avg * ((len_total - iteration) + (self.args.epoch - epoch -1) * len_total)  

                if iteration == len(train_dataloader) - 1:
                    logger.info(
                        ' '.join(
                            ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi},  count: {count}/{max_count}']
                        ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = self.args.count)
                        + ' 2d_loss: {:.8f}, 3d_loss: {:.8f}, 3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}\n'.format(
                            log_2d_losses.avg,
                            log_3d_losses.avg,
                            log_3d_re_losses.avg,
                            pck,
                            log_losses.avg,
                            best_loss,
                            ctime(eta_seconds + end))
                    )
                    writer.add_scalar("Loss/train", log_losses.avg, epoch)

                else:
                    if iteration % self.args.logging_steps == 0:
                        logger.info(
                            ' '.join(
                                ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi},  count: {count}/{max_count}']
                            ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = self.args.count)
                            + ' 2d_loss: {:.8f}, 3d_loss: {:.8f} ,3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                                log_2d_losses.avg,
                                log_3d_losses.avg,
                                log_3d_re_losses.avg,
                                pck,
                                log_losses.avg,
                                best_loss,
                                ctime(eta_seconds + end))
                        )
                    else:
                        logger.debug(
                            ' '.join(
                                ['dataset_length: {len}', 'epoch: {ep}', 'iter: {iter}', '/{maxi},  count: {count}/{max_count}']
                            ).format(len=data_len, ep=epoch, iter=iteration, maxi=len(train_dataloader), count= count, max_count = self.args.count)
                            + ' 2d_loss: {:.8f}, 3d_loss: {:.8f} ,3d_re_loss:{:.8f} ,pck: {:.2f}%, total_loss: {:.8f}, best_loss: {:.8f}, expected_date: {}'.format(
                                log_2d_losses.avg,
                                log_3d_losses.avg,
                                log_3d_re_losses.avg,
                                pck,
                                log_losses.avg,
                                best_loss,
                                ctime(eta_seconds + end))
                        )