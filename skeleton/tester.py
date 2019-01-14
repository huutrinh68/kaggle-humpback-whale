import numpy as np
import pandas as pd
import logging
import torch
import os
import time
import copy

from pathlib import Path
from tqdm import tqdm
from . import drawers
from collections import OrderedDict

class Tester(object):
    """
    Args:
        alchemistic_directory (string):
            The directory which minetorch will use to store everything in
        model (torch.nn.Module):
            Pytorch model optimizer (torch.optim.Optimizer): Pytorch optimizer
        code (str, optional):
            Defaults to "geass". It's a code name of one
            attempt. Assume one is doing kaggle competition and will try
            different models, parameters, optimizers... To keep results of every
            attempt, one should change the code name before tweaking things.
        test_dataloader (torch.utils.data.DataLoader):
            Pytorch dataloader
        hooks (dict, optional):
            Defaults to {}. Define hook functions.
        logging_format ([type], optional):
            Defaults to None. logging format
    """

    def __init__(self, alchemistic_directory, model, code="geass",
                 test_dataloader=None, fold=0,
                 hooks={}, metrics=[], logging_format=None):
        self.alchemistic_directory = alchemistic_directory
        self.code = code
        self.fold = 'FOLD_' + str(fold)
        self.set_logging_config(alchemistic_directory,
                                '{}/{}'.format(self.code, self.fold),
                                logging_format)
        self.models_dir = os.path.join(alchemistic_directory,
                                       '{}/{}'.format(self.code, self.fold),
                                       'models')
        self.model = model
        self.test_dataloader = test_dataloader
        self.hook_funcs = hooks
        self.init_model()
        self.call_hook_func('after_init')
        self.status = 'init'

    def set_logging_config(self, alchemistic_directory, code, logging_format):
        self.log_dir = os.path.join(alchemistic_directory, code)
        log_file = os.path.join(self.log_dir, 'log.txt')
        logging_format = logging_format if logging_format is not None else \
            '%(levelname)s %(asctime)s %(message)s'
        logging.basicConfig(
            filename=log_file,
            format=logging_format,
            datefmt="%m-%d %H:%M:%S",
            level=logging.INFO
        )

    def init_model(self):
        """resume from best checkpoint
        """
        logging.info(self.fold)
        if self.model_file_path('best') is not None:
            checkpoint_path = self.model_file_path('best')
        else:
            checkpoint_path = None
            logging.warning('Could not find checkpoint to load! Stopping')
        logging.info('Checkpoint loaded')
        self.call_hook_func('after_load_checkpoint')

        if checkpoint_path is not None:
            logging.info(f"Start to load checkpoint")
            checkpoint = torch.load(checkpoint_path)
            try:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            except:
                logging.warning(
                    'load checkpoint failed, the state in the '
                    'checkpoint is not matched with the model, '
                    'try to reload checkpoint with unstrict mode')
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info('Checkpoint loaded')

    def call_hook_func(self, name):
        if name not in self.hook_funcs:
            return
        self.hook_funcs[name](self)

    def predict_proba(self):
        """start to test the model
        """
        test_iters = len(self.test_dataloader)
        loader = tqdm(self.test_dataloader, ncols=0)
        loader.set_description('[Fold {:3}]'.format(self.fold))
        predict_probas = []
        targets = []
        for index, data in enumerate(loader):
            predict_proba, target = self.run_test_iteration(index, data, test_iters, loader)
            predict_probas.append(predict_proba.ravel())
            targets.append(target.ravel())
        self.call_hook_func('after_epoch_end')
        return np.vstack(predict_probas), np.vstack(targets)

    def forward(self, images):
        images = images.cuda(non_blocking=True)
        output = self.model(images).sigmoid().cpu().detach().numpy()
        return output

    def run_test_iteration(self, index, data, train_iters, loader):
        self.status = 'test'
        self.call_hook_func('before_test_iteration_start')
        # Predict
        images, targets = data
        outputs = []
        for image in images:
            outputs.append(self.forward(image))
        outputs = np.mean(np.vstack(outputs), axis=0)
        targets = targets.cpu().detach().numpy()
        return outputs, targets

    def model_file_path(self, model_name):
        model_name_path = Path(model_name)
        models_dir_path = Path(self.models_dir)

        search_paths = [
            model_name_path,
            models_dir_path / model_name_path,
            models_dir_path / f'{model_name}.pth.tar',
            models_dir_path / f'epoch_{model_name}.pth.tar',
        ]

        for path in search_paths:
            if path.is_file():
                return path.resolve()
        return None

    # TODO: implement methods below
    def graceful_stop(self):
        """stop train and exist after this epoch
        """
        pass

    def save_and_stop(self):
        """save the model immediately and stop training
        """
        pass

    def check_dir(self):
        """Create directories
        """
        current_dir = self.alchemistic_directory
        current_dir += self.code
        current_dir += self.fold
        current_dir += self.models
        if os.path.isdir(self.current_dir):
            return True
        else:
            return False
