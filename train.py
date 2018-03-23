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
from model.model import ResSal
import data_utils.prepare_data as sal_data
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=64, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='default weight decay')
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
        target = data['gt28'].float()
        if args.cuda:
            # input_var = torch.autograd.Variable(input)
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()
            target = target.cuda()
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        # hidden_maps.register_hook(lambda grad: print(grad.size()))
        output = model(input_var)
        # make_dot(output)
        # output = output.squeeze()
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(output, 1), target_var)
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
        target = data['gt28'].float()
        if args.cuda:
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target, volatile=True).cuda()
            target = target.cuda()
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(output, 1), target_var)
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


def gen_loss_weight(target):
    """generate weight for loss, maybe not necessary"""
    positive_num = torch.sum(target, 1)
    class_num = torch.FloatTensor([target.size(1)]).cuda() if args.cuda else torch.Tensor([target.size(1)])
    negative_num = class_num - positive_num
    weight = torch.div(negative_num, positive_num)
    weight = weight.expand((target.size(0), target.size(1)))
    return torch.mul(weight, target)


def load_pretrained(model, optimizer, fname):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def accuracy(output, target, threshold=0.5):
    """
    Compute precision for multi-label classification part
    accuracy = predict joint target / predict union target
    Use sigmoid function and a threshold to determine the label of output
    :param output: class scores from last fc layer of the model
    :param target: binary list of classes
    :param threshold: threshold for determining class
    :return: accuracy
    """
    sigmoid = torch.sigmoid(output)
    predict = sigmoid > threshold
    target = target > 0
    joint = torch.sum(torch.mul(predict.data, target))
    union = torch.sum(torch.add(predict.data, target) > 0)
    return joint / union


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
