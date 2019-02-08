from __future__ import absolute_import, print_function
"""
Whale data-set for Pytorch
"""
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
Image.MAX_IMAGE_PIXELS = 113250652687104351

import os
from .transforms import CovertBGR
import torchvision.transforms as transforms
from collections import defaultdict

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyData(data.Dataset):
    def __init__(self, root=None, label_csv=None,
                 transform=None, mode='train', loader=default_loader):

        # Initialization data path and train(gallery or query) txt path
        root = os.path.join('/home/tran/workspace/kaggle/new-humpback-whale-identification/input/',
                            mode + '/')

        if transform is None:
            transform_dict = Generate_transform_Dict()['resize']

        # read txt get image path and labels
        file = open(label_csv)
        images_anon = file.readlines()[1:]
        images = []
        labels = []
        indexed_labels = {}

        for img_anon in images_anon:
            [img, label] = img_anon.split(',')
            label = label[:-1]
            if label == 'new_whale': continue
            images.append(img)
            if label not in indexed_labels:
                indexed_labels[label] = len(list(indexed_labels.keys()))
            labels.append(label)

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[indexed_labels[label]].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.indexed_labels = indexed_labels
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        fn, label = self.images[index], self.indexed_labels[self.labels[index]]
        img = self.loader(self.root, fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

class BBLoader():
    def __init__(self, root, crop_margin=0.01):
        bbox_csv = os.path.join(root, 'cropping.csv')
        bbox_df = pd.read_csv(bbox_csv)
        bbox_dict = {}
        for idx, row in bbox_df.iterrows():
            fn = row['Image']
            x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
            bbox_dict[fn] = (x0, y0, x1, y1)
        self.bbox_dict = bbox_dict
        self.crop_margin = crop_margin

    def load(self, root, fn):
        path = os.path.join(root, fn)
        img = Image.open(path)
        size_x, size_y = img.size

        # Crop the image using bbox
        x0, y0, x1, y1 = self.bbox_dict[fn]
        box = x0, y0, x1, y1

        x1 = max(x0, x1)
        y1 = max(y0, y1)
        dx = max(x1 - x0, 0)
        dy = max(y1 - y0, 0)

        margin_x = dx * self.crop_margin
        margin_y = dy * self.crop_margin

        x0 = max(x0 - margin_x, 0)
        y0 = max(y0 - margin_y, 0)
        x1 = min(x1 + margin_x + 1, size_x)
        y1 = min(y1 + margin_y + 1, size_y)
        box = x0, y0, x1, y1

        img = img.crop(box=box)
        return img

class Whale:
    def __init__(self, root=None, origin_width=256, width=224, ratio=0.16, transform=None):
        # Data loading code

        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = '/home/tran/workspace/kaggle/new-humpback-whale-identification/input/'

        train_csv = os.path.join(root, 'train.csv')
        test_csv  = os.path.join(root, 'sample_submission.csv')
        bb_loader = BBLoader(root)
        self.train = MyData(root, label_csv=train_csv, transform=transform_Dict['resize_agg'],
                            mode='train', loader=bb_loader.load)
        self.gallery = MyData(root, label_csv=test_csv, transform=transform_Dict['resize'],
                              mode='test', loader=bb_loader.load)

def Generate_transform_Dict(origin_width=256, width=224, ratio=0.16):
    transform_dict = {}
    transform_dict['resize'] = \
    transforms.Compose([
        transforms.Resize((width, width)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[50/255.0],
                             std=[1.0/255.0]),
    ])
    transform_dict['resize_agg'] = \
    transforms.Compose([
        transforms.Resize((width,width)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.01,
                               saturation=0.01,
                               hue=0.01),
        transforms.RandomRotation(8),
        transforms.RandomResizedCrop(size=512,
                                     scale=(0.95,1)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[50/255.0],
                             std=[1.0/255.0])
])
    return transform_dict

def testWhale():
    data = Whale()
    print(len(data.gallery))
    print(len(data.train))
    for i in range(2):
        print(data.train[i])

if __name__ == "__main__":
    testWhale()


