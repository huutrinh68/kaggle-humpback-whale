from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn

# local import
from sacred import Ingredient
from data import data_ingredient
model_ingredient = Ingredient('model', ingredients=[data_ingredient])


@model_ingredient.config
def cfg():
    model = 'resnet18' # resnet18 / resnet34 / bninception / seres50, default: bninception


@model_ingredient.capture
def load_model(model, data):
    n_classes  = data['n_classes']
    n_channels = data['n_channels']
    if   model == 'resnet34': return _get_net_resnet34(n_channels, n_classes)
    elif model == 'resnet18': return _get_net_resnet18(n_channels, n_classes)
    elif model == 'seres50' : return _get_net_seres50(n_channels, n_classes)
    else: return _get_net_bninception(n_channels, n_classes)


# =======================
# get_net functions
def _get_net_bninception(n_channels, n_classes):
    _model = bninception(pretrained="imagenet")
    new_features = nn.Sequential(*list(_model.children()))
    new_features[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3))
    _model.global_pool = nn.AdaptiveAvgPool2d(1)
    _model.conv1_7x7_s2 = new_features[0]
    _model.last_linear = nn.Sequential(nn.BatchNorm1d(1024),
                                      nn.Dropout(0.5),
                                      nn.Linear(1024, n_classes))
    return _model


def _get_net_resnet34(n_channels, n_classes):
    _model = models.resnet34(pretrained='imagenet')
    _model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                            stride=(2, 2), padding=(3, 3), bias=False)
    _model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = _model.fc.in_features
    _model.fc = nn.Linear(num_ftrs, n_classes)
    return _model


def _get_net_resnet18(n_channels, n_classes):
    _model = models.resnet18(pretrained='imagenet')
    _model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                            stride=(2, 2), padding=(3, 3), bias=False)
    _model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = _model.fc.in_features
    _model.fc = nn.Linear(num_ftrs, n_classes)
    return _model


def _get_net_seres50(n_channels, n_classes):
    _model = pretrainedmodels.models.se_resnet50(pretrained='imagenet')
    _model.layer0.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    _model.avg_pool = nn.AdaptiveAvgPool2d(1) 
    num_ftrs = _model.last_linear.in_features
    _model.last_linear = nn.Linear(num_ftrs, n_classes)
    return _model
