import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_finetune import make_model


# local import
from sacred import Ingredient
from data import data_ingredient
model_ingredient = Ingredient('model', ingredients=[data_ingredient])


@model_ingredient.config
def cfg():
    model    = 'siamese'
    backbone = 'resnet18' # resnet18 / resnet34 / bninception / seres50, default: bninception
    heads    = [32, 64, 128]


@model_ingredient.capture
def load_model(model, backbone, heads):
    if model == 'siamese':
        return Siamese(backbone)
    elif model == 'boosting_siamese':
        return BoostingSiamese(backbone, heads)
    else:
        return Siamese(backbone)

# =====================================
class Siamese(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        backbone = make_model(
            model_name=backbone,
            pretrained=True,
            num_classes=1
        )._features
        backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.name     = 'siamese'
        self.backbone = backbone
        self.feature_dims = 512

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_dims, out_features=1),
        )
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 4), padding=0, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(32, 1), padding=0, stride=1)

    def compute_head_features(self, x, y):
        x1 = x * y
        x2 = x + y
        x3 = (x - y).abs_()
        x4 = (x - y) * (x - y)
        x = torch.cat([x1, x2, x3, x4], 1)
        return x

    def get_score(self, feature_a, feature_b):
        # Make head features
        head_features = self.compute_head_features(feature_a, feature_b)

        head_features = head_features.view(-1, 1, self.feature_dims, 4)
        head_features = F.relu(self.conv1(head_features))
        head_features = head_features.view(-1, 1, 32, self.feature_dims)
        head_features = F.relu(self.conv2(head_features))
        head_features = head_features.view(-1, self.feature_dims)

        score = self.classifier(head_features)
        score = torch.sigmoid(score)
        return score

    def forward(self, x):
        xa, xb = x[0], x[1]
        # Get features
        feature_a = self.get_features(xa)
        feature_b = self.get_features(xb)

        score = self.get_score(feature_a, feature_b)
        return score

    def get_features(self, x):
        x = self.backbone(x)
        return x.reshape(*x.shape[:2], -1).max(-1)[0]

# =====================================
class BoostingSiamese(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        backbone = make_model(
            model_name=backbone,
            pretrained=True,
            num_classes=1
        )._features
        backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.name     = 'boosting_siamese'
        self.backbone = backbone
        self.n_dist   = 3
        self.dims     = 512

        self.heads = []
        for i in range(len(heads)):
            self.heads.append(dict())

            # Saving head info to dict
            self.heads[i]['n_channels']  = heads[i]

            # Create sub-module based on the info
            self.heads[i]['clf']  = nn.Sequential(
                nn.Linear(in_features=self.dims, out_features=1)
            )
            self.heads[i]['conv1'] = nn.Conv2d(1, heads[i],
                                               kernel_size=(1, self.n_dist),
                                               padding=0, stride=1)
            self.heads[i]['conv2'] = nn.Conv2d(1, 1,
                                               kernel_size=(heads[i], 1),
                                               padding=0, stride=1)

            # Adding module to main model
            self.add_module('head_conv1_{}'.format(i), self.heads[i]['conv1'])
            self.add_module('head_conv2_{}'.format(i), self.heads[i]['conv2'])
            self.add_module('head_clf_{}'.format(i),   self.heads[i]['clf'])

        # Calculating the scoring weight for each head
        for i in range(len(heads)):
            tmp = 2.0 / (1 + len(heads))
            self.heads[i]['head_weight'] = tmp
            self.heads[i]['scoring_weight'] = 0
            for k in range(i):
                self.heads[i]['scoring_weight'] = tmp * self.heads[k]['head_weight']

    def get_distance_features(self, input):
        # Get features
        x = self.get_features(input[0])
        y = self.get_features(input[1])

        # Calculate different kinds of distance
        x1 = x * y
        x2 = (x - y).abs_()
        x3 = (x - y) * (x - y)

        # Concat
        dist = torch.cat([x1, x2, x3], 1)

        return dist

    def get_score(self, head, input_features):
        # Make head features
        output = input_features.view(-1, 1, self.dims, self.n_dist)
        output = F.relu(head['conv1'](output))
        output = output.view(-1, 1, head['n_channels'], self.dims)
        output = F.relu(head['conv2'](output))
        output = output.view(-1, self.dims)
        score = head['clf'](output)
        score = torch.sigmoid(score)
        return score

    def forward(self, x):
        dist_features = self.get_distance_features(x)
        # Calculating score for each head then ensemble
        scores = []
        weighted_scores = []
        last_score = 0
        agg_score = 0
        for i, head in enumerate(self.heads):
            score = self.get_score(head, dist_features)
            weighted_score = (1-head['scoring_weight'])*last_score \
                             +  head['scoring_weight']*score
            scores.append(score)
            weighted_scores.append(weighted_score)
        return scores, weighted_scores

    def get_features(self, x):
        x = self.backbone(x)
        x = x.reshape(*x.shape[:2], -1).max(-1)[0]
        return x
