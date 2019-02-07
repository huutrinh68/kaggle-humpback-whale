from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
import torch

def get_net():
    model = models.resnet34(pretrained=True)
    # get the pre-trained weights
    # pretrained_weights = list(model.parameters())
    
    model.conv1 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1.weight = torch.nn.Parameter(torch.cat((pretrained_weights[0], pretrained_weights[0][:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)

    return model
