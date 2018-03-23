"""
A fast deep learning method for saliency detection
=====================================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import torch.nn as nn
import numpy as np
import os
# import shutil
import torchvision.models as models


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DilateConv(nn.Module):
    """
    d_rate: dilation rate
    H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    set kernel size to 3, stride to 1, padding==d_rate ==> spatial size kept
    """

    def __init__(self, d_rate, in_ch, out_ch):
        super(DilateConv, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
            stride=1, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        return self.d_conv(x)
 

class DCPP(nn.Module):
    """
    dilated convolutional pyramid polling
    d_rates: dilation rates as list or tuple,
            number of sub-convs equals to length of d_rates
    out_ch: output channel of one sub dilated convolution
    """

    def __init__(self, d_rates, in_ch, out_ch):
        super(DCPP, self).__init__()
        self.sub_number = len(d_rates)
        self.sub_convs = nn.ModuleList(
            [DilateConv(d_rate, in_ch, out_ch) for d_rate in d_rates])

    def forward(self, x):
        # TODO: find better method to do this
        out_ft = None
        for idx in range(self.sub_number):
            out_ft = self.sub_convs[idx](x) if idx == 0 \
                else torch.cat((out_ft, self.sub_convs[idx](x)), 1)
        return out_ft  # channel of out_ft equals to len(d_rates)*out_ch


class ARM(nn.Module):
    """
    attention residual module
    ft_ch: channel of the input feature map
    tail_block: bool, True if the block has no residual input (last ARM block)
    """

    def __init__(self, ft_ch, tail_block=False, res_ch=32, atten_ch=32):
        super(ARM, self).__init__()
        self.is_tail = tail_block
        self.res_ch = res_ch
        self.atten_ch = atten_ch
        self.cat_ch = self.res_ch + self.atten_ch + 1 if not tail_block else self.atten_ch + 1

        self.ft_conv = nn.Sequential(ConvReLU(ft_ch, out_ch=32, kernel_sz=3))
        self.res_conv = nn.Sequential(ConvReLU(self.cat_ch, self.res_ch, kernel_sz=3))
        self.res_conv2 = nn.Sequential(ConvReLU(self.res_ch, 1, kernel_sz=3))
        self.up_res = nn.Upsample(scale_factor=(1, 1, 2, 2), mode='bilinear')
        self.up_sal = nn.Upsample(scale_factor=(1, 1, 2, 2), mode='bilinear')

    def forward(self, ft, cs, residual=None):
        """
        :param ft: feature from base bone network
        :param cs: coarse saliency prediction from the tail of the network
        :param residual: multi-channel residual
        :return: multi-channel residual, fixed saliency prediction
        """
        attention = self.ft_conv(ft)
        x = torch.cat((attention, cs), 1) if self.is_tail else \
            torch.cat((attention, residual, cs))
        res_out = self.res_conv(x)
        res_map = self.res_conv2(res_out)
        pred_sal = torch.add(cs, res_map)
        return self.up_res(res_out), self.up_sal(pred_sal)


class ResSal(nn.Module):
    def __init__(self):
        super(ResSal, self).__init__()
        self.dilate_ch = 16
        self.dilation_rates = [1, 5, 9, 13]
        self.ft_chs = [512, 256, 128, 56]
        vgg = models.vgg16(pretrained=True)
        self.conv1 = nn.Sequential(*list(vgg.features.children())[0:4])
        self.conv2 = nn.Sequential(*list(vgg.features.children())[4:9])
        self.conv3 = nn.Sequential(*list(vgg.features.children())[9:16])
        self.conv4 = nn.Sequential(*list(vgg.features.children())[16:23])
        self.conv5 = nn.Sequential(*list(vgg.features.children())[23:-1])
        last_ft_ch = 512
        self.dcpp = DCPP(self.dilation_rates, last_ft_ch, self.dilate_ch)
        self.conv6 = nn.Sequential(nn.Conv2d(
            in_channels=len(self.dilation_rates) * self.dilate_ch, out_channels=1, kernel_size=3, padding=1))

        # self.tail_arm = ARM(last_ft_ch, tail_block=True)
        # self.arms = nn.ModuleList([ARM(ft_ch) for ft_ch in self.ft_chs])

        # self.out = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1))
        # self.fc = nn.Sequential(nn.Linear(14 * 14, 2048),
        #                         nn.Linear(2048, 14 * 14))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        sal_coarse = self.dcpp(conv5)
        sal_coarse = self.conv6(sal_coarse)
        # res5, sal5 = self.tail_arm(conv5, sal_coarse)
        # res4, sal4 = self.arms[0](conv4, sal5, res5)
        # res3, sal3 = self.arms[1](conv3, sal4, res4)
        # res2, sal2 = self.arms[2](conv2, sal3, res3)
        # _, sal1 = self.arms[3](conv1, sal2, res2)

        # sal_coarse = self.out(conv5)
        # sal_coarse = sal_coarse.view(sal_coarse.size(0), -1)
        # sal_coarse = self.fc(sal_coarse)
        # sal_coarse = sal_coarse.view(sal_coarse.size(0), 14, 14)
        return sal_coarse
