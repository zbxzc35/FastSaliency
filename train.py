"""
Main code for weakly supervised object localization
===================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
import shutil
import time
import datetime
from model.model import ResSal, load_pretrained
import data_utils.prepare_data as sal_data
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=32, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--step-size', '--ss', default=2, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.5, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-log', default=False,
                    help='disable logging while training')
parser.add_argument('--gpuID', default='0', type=str,
                    help='GPU ID')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID

save_root = './data'
train_data_dir = '/disk2/zhangyu/data/saliency/MSRA10K/'
validation_ratio = 0.2
# vggParas = '/home/zhangyu/data/VGG_imagenet.npy'
# train_dir = '/home/zhangyu/data/tmp/'
check_point_dir = os.path.join(save_root, 'checkpt')
logging_dir = os.path.join(save_root, 'log')
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
if not os.path.isdir(check_point_dir):
    os.mkdir(check_point_dir)
if not os.path.isdir(os.path.join(check_point_dir, 'best_model')):
    os.mkdir(os.path.join(check_point_dir, 'best_model'))


def main():
    global log_file
    log_file = os.path.join(logging_dir, 'log_{}.txt'.format(
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    log_file_npy = os.path.join(logging_dir, 'log_{}.npy'.format(
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = 0
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    net = ResSal()

    if args.cuda:
        net.cuda()

    train_loader, val_loader = prepare_data(train_data_dir, validation_ratio)

    # net = torch.nn.DataParallel(net).cuda()
    # optimizer = torch.optim.SGD(net.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))

    train_loss = []
    train_loss_detail = []
    val_loss = []
    for epoch in range(args.start_epoch, args.epochs):
        tr_avg_loss, tr_detail_loss = train(train_loader, net, optimizer, epoch)
        val_avg_loss = validation(val_loader, net)

        adjust_lr(optimizer=optimizer, epoch=epoch, step_size=args.step_size, gamma=args.gamma)

        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss
        val_loss.append(val_avg_loss)
        np.save(log_file_npy, {'train_loss': train_loss,
                               'train_loss_detail': train_loss_detail,
                               'val_loss': val_loss})

        # Save checkpoint
        save_file = os.path.join(
            check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch+1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)
        # if(is_best):
        #     tmp_file_name = os.path.join(check_point_dir, 'best_model',
        #         'best_checkpoint_epoch{}.pth.tar'.format(best_epoch))
        #     if os.path.isfile(tmp_file_name):
        #         os.remove(tmp_file_name)
        #     best_epoch = epoch + 1
        #     shutil.copyfile(save_file, os.path.join(
        #         check_point_dir, 'best_model',
        #         'best_checkpoint_epoch{}.pth.tar'.format(best_epoch)))


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    epoch_loss = []
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare input
        input = data['img'].float()
        target28 = data['gt28'].float()
        target56 = data['gt56'].float()
        target112 = data['gt112'].float()
        target224 = data['gt224'].float()
        if args.cuda:
            input_var = torch.autograd.Variable(input).cuda()
            target28_var = torch.autograd.Variable(target28).cuda()
            target56_var = torch.autograd.Variable(target56).cuda()
            target112_var = torch.autograd.Variable(target112).cuda()
            target224_var = torch.autograd.Variable(target224).cuda()
            # target = target.cuda()
        else:
            pass
            # Not implemented
            # input_var = torch.autograd.Variable(input)
            # target_var = torch.autograd.Variable(target)

        # compute output
        # hidden_maps.register_hook(lambda grad: print(grad.size()))
        pred_224, pred_112, pred_56, pred_28_c4, pred_28_c5, pred_28_cs = model(input_var)
        # make_dot(output)
        # output = output.squeeze()
        loss_28_cs = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_cs, 1), target28_var)
        loss_28_c5 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_c5, 1), target28_var)
        loss_28_c4 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_c4, 1), target28_var)
        loss_56 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_56, 1), target56_var)
        loss_112 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_112, 1), target112_var)
        loss_224 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_224, 1), target224_var)
        loss = loss_28_cs + loss_28_c5 + loss_28_c4 + loss_56 + loss_112 + loss_224
        if args.cuda:
            loss = loss.cuda()

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        epoch_loss.append(loss.data[0])
        batch_time.update(time.time() - end)
        end = time.time()

        # display and logging
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}][{1}/{2}] '.format(epoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Data {data_time.val:.3f} (avg:{data_time.avg:.3f}) '.format(data_time=data_time) + \
                   'Loss {loss.val:.4f} (avg:{loss.avg:.4f}) '.format(loss=losses)
            print(info)
            if not args.no_log:
                with open(log_file, 'a+') as f:
                    f.write(info + '\n')

        # output.register_hook(lambda grad: print(grad))
        # loss.register_hook(lambda  loss: print(loss))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)
        optimizer.step()

    # return loss, accuracy for recording and plotting
    return losses.avg, epoch_loss


def validation(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        input = data['img'].float()
        target28 = data['gt28'].float()
        target56 = data['gt56'].float()
        target112 = data['gt112'].float()
        target224 = data['gt224'].float()
        if args.cuda:
            input_var = torch.autograd.Variable(input).cuda()
            target28_var = torch.autograd.Variable(target28).cuda()
            target56_var = torch.autograd.Variable(target56).cuda()
            target112_var = torch.autograd.Variable(target112).cuda()
            target224_var = torch.autograd.Variable(target224).cuda()
        else:
            pass
            # Not implemented

        # compute output
        # output = model(input_var)
        # loss = F.binary_cross_entropy_with_logits(torch.squeeze(output, 1), target_var)
        pred_224, pred_112, pred_56, pred_28_c4, pred_28_c5, pred_28_cs = model(input_var)
        # make_dot(output)
        # output = output.squeeze()
        loss_28_cs = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_cs, 1), target28_var)
        loss_28_c5 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_c5, 1), target28_var)
        loss_28_c4 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_28_c4, 1), target28_var)
        loss_56 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_56, 1), target56_var)
        loss_112 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_112, 1), target112_var)
        loss_224 = F.binary_cross_entropy_with_logits(torch.squeeze(pred_224, 1), target224_var)
        loss = loss_28_cs + loss_28_c5 + loss_28_c4 + loss_56 + loss_112 + loss_224
        if args.cuda:
            loss = loss.cuda()

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = 'Test: [{0}/{1}] '.format(i, len(val_loader)) + \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(loss=losses)
            print(info)
            if not args.no_log:
                with open(log_file, 'a+') as f:
                    f.write(info + '\n')

    return losses.avg


def adjust_lr(optimizer, epoch, step_size, gamma):
    # Adjust learning rate every step_size epochs, lr_policy: step
    if (epoch + 1) > 0 and (epoch + 1) % step_size == 0:
        pre_lr = optimizer.param_groups[0]['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma  # gamma
        print("Adjust learning rate, pre_lr: {}, current_lr: {}".format(pre_lr, optimizer.param_groups[0]['lr']))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def prepare_data(data_dir, val_ratio):
    # prepare dataloader for training and validation
    dataset = sal_data.SalData(data_dir)

    data_size = len(dataset)
    train_size = int(data_size * (1 - val_ratio))
    train_ids = list(range(data_size))
    np.random.shuffle(train_ids)
    train_sampler = sampler.SubsetRandomSampler(train_ids[:train_size])
    val_sampler = sampler.SubsetRandomSampler(train_ids[train_size:])

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.workers, drop_last=True)
    return train_loader, val_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
