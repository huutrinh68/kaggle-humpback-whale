import random
import torch
import warnings
import os
import sys
import json
import numpy as np
import pandas as pd
from skeleton.tester import Tester
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.commands import print_config
from tqdm import tqdm
warnings.filterwarnings('ignore')

# local import
from optimizer import optimizer_ingredient, load_optimizer
from criterion import criterion_ingredient, load_loss, f1_macro_aggregator
from model import model_ingredient, load_model
from data import data_ingredient, create_feature_loader
from path import path_ingredient, prepair_dir

ex = Experiment('Test', ingredients=[model_ingredient,      # model
                                     optimizer_ingredient,  # optimizer
                                     data_ingredient,       # data
                                     path_ingredient,       # path
                                     criterion_ingredient]) # criterion

# ex.observers.append(MongoObserver.create(db_name='humpback_whale'))
# ex.observers.append(FileStorageObserver.create('exp_logs/experiments'))
ex.captured_out_filter = apply_backspaces_and_linefeeds
config_dict = None

@ex.config
def cfg():
    max_epochs = None #dummy
    resume = None #dummy
    exp_id = None
    threshold = 0.99
    debug = False
    comment = ''
    tmp0, tmp1 = None, None

@ex.named_config
def reload():
    exp_id = sys.argv[-1].split('=')[-1]
    if exp_id == 'reload_cfg':
        print('The id of the experiment is not inputted! Abort!')
    elif not os.path.isfile('exp_logs/experiments/{}/config.json'.format(exp_id)):
        print('exp_logs/experiments/{}/config.json'.format(exp_id),
              'not found! Use current config.')
    else:
        for tmp0, tmp1 in get_config_dict(exp_id).items():
            locals()[tmp0] = tmp1

def get_config_dict(exp_id):
    config_path = 'exp_logs/experiments/{}/config.json'.format(exp_id)
    with open(config_path, 'r') as f:
        json_str = f.read()
    config_dict = json.loads(json_str)
    return config_dict

@ex.capture
def init(_run, seed, path):
    prepair_dir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

    print('='*50)
    print_config(_run)
    print('='*50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using', torch.cuda.device_count(), 'gpus!')

    return device

@ex.main
def main(path):
    # Model
    device = init()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(load_model()).to(device)
    else:
        model = load_model().to(device)

    train_feature_loader, test_feature_loader = create_feature_loader()
    data_iter = iter(train_feature_loader)
    example_batch = next(data_iter)

    # Features generating
    train_feature_gen = Tester(
        alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
        code = get_dir_name(),
        model=model,
        test_dataloader=train_feature_loader,
    )


@ex.capture
def get_dir_name(model, optimizer, data, path, criterion, seed, comment):
    name = model['model'] + '_' + model['backbone']
    name += '_' + optimizer['optimizer'] + '_' + str(optimizer['lr'])
    name += '_' + criterion['loss']
    name += '_' + str(seed)
    name += '_' + str(comment)
    print('Experiment code:', name)
    return name


if __name__ == '__main__':
    ex.run_commandline()
