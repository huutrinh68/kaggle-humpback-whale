import os
import torch

from models.model import*
from config import config


def load_model():
    ensemble_model = []
    # path1 = "path_to_model_1"
    # model1 = get_net()
    # model1.load_state_dict(torch.load(path1)["state_dict"]) 
    # model1.cuda()
    # # add model
    # ensemble_model.append(model1)

    ensemble_model.append(1)
    ensemble_model.append(2)

    return ensemble_model


def ensemble():
    print("This is ensemble_model")
    ensemble_model = load_model()
    print(ensemble_model)


if __name__ == '__main__':
    ensemble()