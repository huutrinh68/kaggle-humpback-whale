import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cnn_finetune import make_model

class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim, eps=1e-5)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        # x = self.bn(x)
        # x = F.relu(x, inplace=True)
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x

class FineTune(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        backbone = make_model(
            model_name=backbone,
            pretrained=True,
            num_classes=1
        )._features
        self.backbone = backbone
        self.feature_dims = 512
        self.dim = dim
        self.classifier = Embedding(in_dim  = self.feature_dims,
                                    out_dim = self.dim,
                                    normalized = True)

    def forward(self, input):
        return self.classifier(self.get_features(input))

    def get_features(self, x):
        x = self.backbone(x)
        dim = 1
        for d in x.shape[2:]: dim *= d
        return x.resize(*x.shape[:2], dim).max(-1)[0]

def Fine_Tune(backbone='resnet18', dim=512, pretrained=True, model_path=None):
    model = FineTune(backbone, dim=dim)
    return model
