"""
Data loading script for saliency detection with multi-scale supervision
=====================================

*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import os
import numpy as np
# from bs4 import BeautifulSoup
from torch.utils.data import Dataset
import skimage
from skimage import io
from skimage.transform import resize
import random
import torchvision.transforms as transforms


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        return None
    else:
        ext = allfiles[0].split('.')[-1]
        filelist = [fname.replace(''.join(['.', ext]), '') for fname in allfiles]
        return ext, filelist


class Augment(object):
    """
    Augment image as well as target(image like array, not box)
    augmentation include Crop Pad and Filp
    """
    def __init__(self, size_h=15, size_w=15, padding=None, p_flip=None):
        super(Augment, self).__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.padding = padding
        self.p_flip = p_flip

    def get_params(self, img):
        im_sz = img.shape[:2]
        row1 = random.randrange(self.size_h)
        row2 = -random.randrange(self.size_h)-1  # minus 1 to avoid row1==row2==0
        col1 = random.randrange(self.size_w)
        col2 = -random.randrange(self.size_w)-1
        if row1 - row2 >= im_sz[0] or col1 - col2 >= im_sz[1]:
            raise ValueError("Image size too small, please choose smaller crop size")
        padding = None
        if self.padding is not None:
            padding = random.randint(0, self.padding)
        flip_method = None
        if self.p_flip is not None and random.random() < self.p_flip:
            if random.random() < 0.5:
                flip_method = 'lr'
            else:
                flip_method = 'ud'
        return row1, row2, col1, col2, flip_method, padding

    def transform(self, img, row1, row2, col1, col2, flip_method, padding=None):
        """img should be 2 or 3 dimensional numpy array"""
        img = img[row1:row2, col1:col2, :] if len(img.shape) == 3 else img[row1:row2, col1:col2]
        if padding is not None:  # TODO: not working yet, fix it later
            pad = transforms.Pad(padding)
            topil = transforms.ToPILImage()
            img = pad(topil(img))
            img = np.array(img)
        if flip_method is not None:
            if flip_method == 'lr':
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        return img

    def __call__(self, img, target):
        """img and target should have the same spatial size"""
        paras = self.get_params(img)
        img = self.transform(img, *paras)
        target = self.transform(target, *paras)
        return img, target


class SalData(Dataset):
    """Dataset for saliency detection"""
    def __init__(self, dataDir, augmentation=True):
        super(SalData, self).__init__()
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError('Please put your images in folder \'images\' and GT in \'GT\'')
        self.dataDir = dataDir
        _, self.imgList = fold_files(os.path.join(dataDir, 'images'))
        self.augmentation = augmentation
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgName = self.imgList[idx]
        img = skimage.img_as_float(io.imread(os.path.join(self.dataDir, 'images', imgName + '.jpg')))
        gt = skimage.img_as_float(io.imread(os.path.join(self.dataDir, 'GT', imgName + '.png')))
        if self.augmentation is True:
            aug = Augment(size_h=15, size_w=15, p_flip=0.5)
            img, gt = aug(img, gt)
        img = resize(img, (224, 224), mode='reflect')
        gt = resize(gt, (224, 224), mode='reflect')
        gt112 = resize(gt, (112, 112), mode='reflect')
        gt56 = resize(gt, (56, 56), mode='reflect')
        gt28 = resize(gt, (28, 28), mode='reflect')
        # Normalize image
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        sample = {'img': img, 'gt224': gt, 'gt112': gt112, 'gt56': gt56, 'gt28': gt28}
        return sample
