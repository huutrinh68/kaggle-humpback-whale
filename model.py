from cnn_finetune import make_model
import torch.nn as nn

# local import
from sacred import Ingredient
from data import data_ingredient
model_ingredient = Ingredient('model', ingredients=[data_ingredient])


@model_ingredient.config
def cfg():
    backbone = 'resnet18' # resnet18 / resnet34 / bninception / seres50, default: bninception

@model_ingredient.capture
def load_model(backbone, data):
    return Siamese(backbone,
                   image_size=(data['image_size'],
                               data['image_size']))

def compute_head_features(x, y):
    x1 = x * y
    x2 = x + y
    x3 = (x - y).abs_()
    x4 = (x - y) * (x - y)
    x = torch.cat([x1, x2, x3, x4], 1)
    return x

class Siamese(nn.Module):
    def __init__(self, backbone='resnet18', image_size=(224, 224)):
        super().__init__()
        backbone = make_model(
            model_name=backbone,
            pretrained=True,
            num_classes=1
        )._features
        backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone = backbone
        self.feature_dims = 512

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_dims, out_features=1),
        )
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 4), padding=0, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(32, 1), padding=0, stride=1)

    def forward(self, x):
        xa, xb = x[0], x[1]
        # Get features
        feature_a = self.get_features(xa)
        feature_b = self.get_features(xb)

        return (feature_a, feature_b)
        # score = self.get_score(feature_a, feature_b)
        # return score, feature_a, feature_b

    def get_features(self, x):
        x = self.backbone(x)
        return x.reshape(*x.shape[:2], -1).max(-1)[0]

    # def get_score(self, feature_a, feature_b):
    #     # Make head features
    #     head_features = compute_head_features(feature_a, feature_b)

    #     head_features = head_features.view(-1, 1, self.feature_dims, 4)
    #     head_features = F.relu(self.conv1(head_features))
    #     head_features = head_features.view(-1, 1, 32, self.feature_dims)
    #     head_features = F.relu(self.conv2(head_features))
    #     head_features = head_features.view(-1, self.feature_dims)

    #     score = self.classifier(head_features)
    #     score = torch.sigmoid(score)
    #     return score
