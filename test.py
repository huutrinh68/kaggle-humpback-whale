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

@ex.capture
def optimize_threshold(predict_proba, target, thresholds):
    best_thredshold = 0
    best_f1 = -np.inf
    for threshold in tqdm(thresholds, ncols=0):
        predict = predict_proba > threshold
        macro_f1_score = f1_score(target, predict, average='macro')
        if macro_f1_score > best_f1:
            best_f1 = macro_f1_score
            best_threshold = threshold
    logging.info('best_threshold is {} with val f1 score = {}'.format(best_threshold, best_f1))
    return best_threshold

@ex.capture
def make_prediction(predict_proba, threshold):
    predicts = predict_proba > threshold
    submissions = []
    for i in range(predicts.shape[0]):
        _pred = predicts[i, :]
        pred = list(np.where(_pred.ravel() > threshold)[0].ravel())
        if len(pred) == 0: pred = [np.argmax(_pred.ravel())]
        submissions.append(pred)
    return submissions

@ex.capture
def voting(preds, nvoters):
    predict_set = set(preds)
    final_pred = [v for v in predict_set if preds.count(v) > nvoters / 2]
    return final_pred

@ex.capture
def preds_to_str(preds):
    return ' '.join([str(v) for v in preds])

@ex.capture
def voting_ensemble(ensemble_submissions):
    n_model = len(ensemble_submissions)
    n_pred  = len(ensemble_submissions[0])
    final_preds = []
    for k in range(n_pred):
        preds = []
        for i in range(n_model):
            preds += ensemble_submissions[i][k]
        final_preds.append(preds_to_str(voting(preds, n_model)))
    return final_preds

@ex.main
def main(_log, code, data, path, seed, threshold, find_threshold, debug):
    device = init()
    fold_submissions = []
    for fold in range(data['n_fold']):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(load_model()).to(device)
        else:
            model = load_model().to(device)

        test_loader, val_loader = create_test_loader(fold=fold, n_fold = data['n_fold'],
                                                     seed=seed, debug=debug)

        test_tester = Tester(
            alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
            code = code, fold=fold, model=model, test_dataloader=test_loader,
        )

        val_tester = Tester(
            alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
            code = code, fold=fold, model=model, test_dataloader=val_loader,
        )

        if debug:
            test_predict_proba, _           = test_tester.predict_proba()
            val_predict_proba, val_targets  = val_tester.predict_proba()
        else:
            try:
                test_predict_proba, _           = test_tester.predict_proba()
                val_predict_proba, val_targets  = val_tester.predict_proba()
            except Exception as e:
                _log.error('Unexpected exception! %s', e)

        if find_threshold == False:
            thresholds = [threshold]
        else:
            thresholds = np.linspace(0.1, 0.5, 50)

        best_threshold = optimize_threshold(val_predict_proba, val_targets, thresholds)
        fold_submissions.append(make_prediction(test_predict_proba, best_threshold))
        logging.info('='*50)

    final_submissions = voting_ensemble(fold_submissions)
    submit_df = pd.read_csv(path['root'] + path['sample_submission'])
    if debug is True: submit_df = submit_df[:100]
    submit_df['Predicted'] = final_submissions
    submit_df.to_csv(path['root'] + path['submit'] + code + '.csv', index=False)

@ex.capture
def get_dir_name(model, optimizer, data, path, criterion, seed, comment):
    name = model['model']
    name += '_' + optimizer['optimizer'] + '_' + str(optimizer['lr'])
    name += '_' + criterion['loss'] + '_' + criterion['weight']
    name += '_' + str(seed)
    name += '_' + str(comment)
    logging.info('Experiment code: {}'.format(name))
    return name

if __name__ == '__main__':
    ex.run_commandline()
