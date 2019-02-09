import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cnn_finetune import make_model

# class Embedding(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
#         super(Embedding, self).__init__()
#         self.bn = nn.BatchNorm2d(in_dim, eps=1e-5)
#         self.linear_1 = nn.Linear(in_dim, 2048)
#         self.linear_2 = nn.Linear(2048, in_dim)
#         self.dropout = dropout
#         self.normalized = normalized

#     def forward(self, x):
#         # x = self.bn(x)
#         # x = F.relu(x, inplace=True)
#         if self.dropout is not None:
#             x = nn.Dropout(p=self.dropout)(x, inplace=True)
#         x = self.linear_1(x)
#         x = F.relu(x, inplace=True)
#         if self.dropout is not None:
#             x = nn.Dropout(p=self.dropout)(x, inplace=True)
#         x = self.linear_2(x)
#         if self.normalized:
#             norm = x.norm(dim=1, p=2, keepdim=True)
#             x = x.div(norm.expand_as(x))
#         return x

class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, normalized=True):
        super(Embedding, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_dim, eps=1e-5)
        self.bn2 = nn.BatchNorm1d(512, eps=1e-5)
        self.linear1 = nn.Linear(in_features=in_dim, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.bn1(x)
        x = nn.Dropout(p=self.dropout, inplace=True)(x)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = nn.Dropout(p=self.dropout, inplace=True)(x)
        x = self.linear2(x)
        return x

def make_classifier(in_dim, out_dim):
    return Embedding(in_dim, out_dim, normalized=True)

def Fine_Tune(backbone='resnet18', dim=512, pretrained=True, model_path=None):
    model = make_model(model_name=backbone, pretrained=True, num_classes=dim,
                       classifier_factory=make_classifier)
    return model
