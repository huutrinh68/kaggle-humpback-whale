import random
import torch
import warnings
import logging
import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from tqdm import tqdm
from skeleton.tester import Tester

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.commands import print_config

# local import
from optimizer import optimizer_ingredient, load_optimizer
from criterion import criterion_ingredient, load_loss, f1_macro_aggregator
from model import model_ingredient, load_model
from data import data_ingredient, create_test_loader
from path import path_ingredient, prepair_dir
from utils import sigmoid

ex = Experiment('Test', ingredients=[model_ingredient,      # model
                                       data_ingredient,       # data
                                       path_ingredient])       # path
ex.observers.append(MongoObserver.create(db_name='human_protein'))
ex.observers.append(FileStorageObserver.create('exp_logs/experiments'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    code = None
    seed=2050
    threshold = 0.2
    resume = True
    debug = False
    comment = ''
    find_threshold = True

@ex.capture
def init(_run, seed, path):
    prepair_dir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

    logging.info('='*50)
    print_config(_run)
    logging.info('='*50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using {} gpus!'.format(torch.cuda.device_count()))

    return device

@ex.main
def main(_log, code, data, path, seed, threshold, find_threshold, debug):
    device = init()
    fold_submissions = []
    for fold in range(data['n_fold']):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(load_model()).to(device)
        else:
            model = load_model().to(device)

    # Data
    train_siamese_loader, val_siamese_loader = create_siamese_loader()

    # Loss function
    loss_func = load_loss()

    test_tester = Tester(
        alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
        code = code, fold=fold, model=model, test_dataloader=test_loader,
    )

@ex.capture
def get_dir_name(model, optimizer, data, path, criterion, seed, comment):
    name = model['backbone']
    name += '_' + optimizer['optimizer'] + '_' + str(optimizer['lr'])
    name += '_' + criterion['loss']
    name += '_' + str(seed)
    name += '_' + str(comment)
    print('Experiment code:', name)
    return name

if __name__ == '__main__':
    ex.run_commandline()
