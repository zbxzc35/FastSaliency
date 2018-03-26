"""
Test code for weakly supervised object localization
===================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import os
from os.path import join as pjoin
from skimage.transform import resize
import numpy as np
from model.model import ResSal, load_pretrained
import skimage.io as io
import time
import skimage
import warnings

useGPU = True

gpuID = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

data_root = './data'
save_root = pjoin(data_root, 'results')
datasets_dir = '/disk2/zhangyu/data/saliency/'
test_datasets = ['ECSSD', 'PASCAL-S', 'BSD', 'DUT-O']
check_point_dir = pjoin(data_root, 'checkpt')
ckpt_epoch = 30
ckpt_file = pjoin(check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(ckpt_epoch))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def main():
    net = ResSal()
    if useGPU is True:
        net = net.cuda()
    net, _ = load_pretrained(model=net, fname=ckpt_file)

    for dataset in test_datasets:
        sal_save_dir = pjoin(save_root, dataset)
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = pjoin(datasets_dir, dataset, 'images')
        img_list = os.listdir(img_dir)
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for img_name in img_list:
                img = skimage.img_as_float(io.imread(pjoin(img_dir, img_name)))
                h, w = img.shape[:2]
                img = resize(img, (224, 224))
                img = np.transpose((img - mean) / std, (2, 0, 1))
                img = torch.unsqueeze(torch.FloatTensor(img), 0)
                input_var = torch.autograd.Variable(img)
                if useGPU is True:
                    input_var = input_var.cuda()
                predict, _, _, _, _, _ = net(input_var)
                predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
                predict = predict.data.cpu().numpy()
                predict = resize(predict, (h, w))
                save_file = pjoin(sal_save_dir, img_name.strip('.jpg') + '.png')
                io.imsave(save_file, predict)
        duration = time.time() - start_time
        print('Dataset: {}, {} images, test time: {}, speed: {}fps'.format(dataset, len(img_list), duration, len(img_list)/duration))


if __name__ == '__main__':
    main()
