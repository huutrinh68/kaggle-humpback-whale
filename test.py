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
from data import data_ingredient, create_image_loader, create_feature_loader
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
            if tmp0 != 'threshold':
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

def gen_feature(model, loader):
    features = []
    for index, images in tqdm(enumerate(loader)):
        images = images.cuda(non_blocking=True)
        feature = model.get_features(images).detach().cpu().numpy()
        features.append(feature)
    features = np.concatenate(features, axis=0)
    return features

def gen_score(model, train_loader, test_loader):
    scores = np.zeros((len(test_loader.dataset),
                       len(train_loader.dataset)))
    print(scores.shape)
    i = 0
    for test_feature in tqdm(test_loader):
        k = 0
        for train_feature in train_loader:
            x = test_feature
            y = train_feature

            n_x = x.shape[0]
            n_y = y.shape[0]
            n_dim = x.shape[1]

            x = x.repeat(1,1,n_y).reshape(-1,n_dim).cuda()
            y = y.repeat(1,1,n_x).reshape(-1,n_dim).cuda()

            score = model.get_score(x, y).detach().cpu().numpy()
            score = score.reshape((n_x, n_y))
            scores[i:i+n_x, k:k+n_y] = score

            k += n_y
        i += n_x

    return scores

def prepare_submit(scores, train_dataset, test_dataset,
                   threshold, filename):
    print(scores.shape)
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(test_dataset.data)):
            d = dict()
            a = scores[i, :]
            for j in list(np.argsort(a)):
                t = train_dataset.data[j]
                w = train_dataset.t2w[t]
                if w not in d: d[w]  = 1
                else:          d[w] += 1
                if len(d) == 5:
                    break
            d = d.items()
            d = sorted(d, key=lambda x:x[1], reverse=True)
            ws = [w for w,c in d]
            f.write(p + ',' + ' '.join(ws) + '\n')

@ex.main
def main(path, data, threshold):
    # Prepare
    device = init()
    model = load_model().cuda()
    train_image_loader, test_image_loader = create_image_loader()

    # Load features
    train_features = torch.from_numpy(gen_feature(model, train_image_loader))
    test_features  = torch.from_numpy(gen_feature(model, test_image_loader))

    # Calculate scores
    train_feature_loader, test_feature_loader = create_feature_loader(train_features,
                                                                      test_features)
    scores = gen_score(model, train_feature_loader, test_feature_loader)
    prepare_submit(scores,
                   train_image_loader.dataset,
                   test_image_loader.dataset,
                   threshold,
                   path['submit'] + get_dir_name() + '_' + str(threshold))

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
