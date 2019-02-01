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

class Trainer(object):
    """The heart of minetorch

    Args:
        alchemistic_directory (string):
            The directory which minetorch will use to store everything in
        model (torch.nn.Module):
            Pytorch model optimizer (torch.optim.Optimizer): Pytorch optimizer
        loss_func (function):
            A special hook function to compute loss, the function receive 2 variable:
            * Trainer: the trainer object
            * Data: Batch data been yield by the loader
            return value of the hook function should be a float number of the loss
        code (str, optional):
            Defaults to "geass". It's a code name of one
            attempt. Assume one is doing kaggle competition and will try
            different models, parameters, optimizers... To keep results of every
            attempt, one should change the code name before tweaking things.
        train_dataloader (torch.utils.data.DataLoader):
            Pytorch dataloader
        val_dataloader (torch.utils.data.DataLoader, optional):
            Defaults to None, if no validation dataloader is provided, will skip validation
        resume (bool, optional):
            Defaults to True. Resume from last training, could be:
            * True: resume from the very last epochs
            * String: resume from the specified epochs
                          etc. `34`, `68` `best`
        eval_stride (int, optional):
            Defaults to 1. Validate every `eval_stride` epochs
        persist_stride (int, optional):
            Defaults to 1.
            Save model every `persist_stride` epochs
        drawer (minetorch.Drawer or string, optional):
            Defaults to matplotlib.
            If provide, Trainer will draw training loss and validation loss
            curves, could be `tensorboard` or self implemented Drawer object
        hooks (dict, optional):
            Defaults to {}. Define hook functions.
        max_epochs ([type], optional):
            Defaults to None. How many epochs to train, None means unlimited.
        logging_format ([type], optional):
            Defaults to None. logging format
    """

    def __init__(self, alchemistic_directory, model, optimizer, loss_func,
                 code="geass", train_dataloader=None, val_dataloader=None,
                 resume=True, eval_stride=1, persist_stride=1, fold=0,
                 drawer=None, hooks={}, max_epochs=None, metrics=[],
                 logging_format=None):
        self.alchemistic_directory = alchemistic_directory
        self.code = code
        self.fold = 'FOLD_' + str(fold)
        self.create_dirs()
        self.set_logging_config(alchemistic_directory,
                                '{}/{}'.format(self.code, self.fold),
                                logging_format)
        self.create_drawer(drawer)
        self.models_dir = os.path.join(alchemistic_directory,
                                       '{}/{}'.format(self.code, self.fold),
                                       'models')

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_func = loss_func
        self.resume = resume
        self.eval_stride = eval_stride
        self.persist_stride = persist_stride

        self.lowest_train_loss = float('inf')
        self.lowest_val_loss = float('inf')
        self.current_epoch = 0
        self.hook_funcs = hooks
        self.max_epochs = max_epochs

        self.metrics = [('loss', mean_aggregator(self.loss_func))] + metrics
        self.metrics = OrderedDict((metric, {'train': copy.deepcopy(aggregator),
                                             'val': copy.deepcopy(aggregator)})
                                   for metric, aggregator in self.metrics)
        self.cache = OrderedDict((metric, {'train': [],'val': []})
                                 for metric in self.metrics)

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

    def create_drawer(self, drawer):
        if drawer == 'tensorboard':
            self.drawer = drawers.TensorboardDrawer(
                self.alchemistic_directory, self.code)
        elif drawer == 'matplotlib':
            self.drawer = drawers.MatplotlibDrawer(
                self.alchemistic_directory, self.code)
        else:
            self.drawer = drawer

    def init_model(self):
        """resume from some checkpoint
        """
        logging.info(self.fold)
        if self.resume is True:
            # resume from the newest model
            if self.model_file_path('latest') is not None:
                checkpoint_path = self.model_file_path('latest')
            else:
                checkpoint_path = None
                logging.warning('Could not find checkpoint to resume, '
                                'train from scratch')
        elif isinstance(self.resume, str):
            checkpoint_path = self.model_file_path(self.resume)
        elif isinstance(self.resume, int):
            checkpoint_path = self.model_file_path(str(self.resume))
        else:
            checkpoint_path = None

        if self.resume is not True and self.resume and checkpoint_path is None:
            # user has specified a none existed model, should raise a error
            raise Exception(f"Could not find model {self.resume}")

        if checkpoint_path is not None:
            logging.info(f"Start to load checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.current_epoch = checkpoint['epoch']
            self.lowest_train_loss = checkpoint['lowest_train_loss']
            self.lowest_val_loss = checkpoint['lowest_val_loss']
            self.cache = checkpoint['cache']

            try:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            except:
                logging.warning(
                    'load checkpoint failed, the state in the '
                    'checkpoint is not matched with the model, '
                    'try to reload checkpoint with unstrict mode')
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if (self.drawer is not None) and ('drawer_state' in checkpoint):
                self.drawer.set_state(checkpoint['drawer_state'])
            logging.info('Checkpoint loaded')
            self.call_hook_func('after_load_checkpoint')

    def call_hook_func(self, name):
        if name not in self.hook_funcs:
            return
        self.hook_funcs[name](self)

    def train(self):
        """start to train the model
        """
        while True:
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.call_hook_func('before_quit')
                logging.info('exceed max epochs, quit!\n')
                break

            self.call_hook_func('before_epoch_start')
            self.current_epoch += 1

            self.model.train()
            train_iters = len(self.train_dataloader)
            val_iters = len(self.val_dataloader)

            # Train epoch
            total_train_loss = 0
            train_loss = np.inf
            val_loss = np.inf
            loader = tqdm(self.train_dataloader, ncols=0)
            loader.set_description('[Epoch {:3d}]'.format(self.current_epoch))
            for index, data in enumerate(loader):
                # Loss
                batch_loss = self.run_train_iteration(index, data, train_iters)
                total_train_loss += batch_loss
                metrics_str = ''
                for metric in self.metrics:
                    metrics_str += ' |'
                    metrics_str += ' train_{}: {:.5f}'.format(metric, self.metrics[metric]['train'].mean())
                    metrics_str += ' val_{}: {:.5f}'.format(metric, self.metrics[metric]['val'].mean())

                loader.set_postfix_str(metrics_str)
                if index == len(loader)-1:
                    time.sleep(0.2)
                    train_loss = total_train_loss / train_iters
                    # Eval epoch
                    total_val_loss = 0
                    if self.val_dataloader is not None:
                        val_iters = len(self.val_dataloader)
                        with torch.set_grad_enabled(False):
                            self.model.eval()
                            for index, data in enumerate(self.val_dataloader):
                                total_val_loss += self.run_val_iteration(index, data, val_iters)
                        val_loss = total_val_loss / val_iters

                    metrics_str = ''
                    for metric in self.metrics:
                        metrics_str += ' |'
                        metrics_str += ' train_{}: {:.5f}'.format(metric, self.metrics[metric]['train'].mean())
                        metrics_str += ' val_{}: {:.5f}'.format(metric, self.metrics[metric]['val'].mean())

                    loader.set_postfix_str(metrics_str)
                    logging.debug(metrics_str)

            for metric in self.metrics:
                self.cache[metric]['train'].append(self.metrics[metric]['train'].mean())
                self.cache[metric]['val'].append(self.metrics[metric]['val'].mean())

            if self.drawer is not None:
                self.drawer.scalars(
                    {'train': train_loss, 'val': val_loss}, 'loss'
                )

            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss

            if val_loss < self.lowest_val_loss:
                logging.debug(
                    'current val loss {:.5f} is lower than lowest {:.5f}, '
                    'persist this model as best one'.format(
                        val_loss, self.lowest_val_loss))

                self.lowest_val_loss = val_loss
                self.persist('best')
            self.persist('latest')

            if not self.current_epoch % self.persist_stride:
                self.persist('epoch_{}'.format(self.current_epoch))

            self.call_hook_func('after_epoch_end')

    def to_cuda(self, inputs):
        if type(inputs) is list:
            inputs = [input.cuda(non_blocking=True)
                      for input in inputs]
        else:
            inputs.cuda(non_blocking=True)
        return inputs

    def forward(self, images):
        images = self.to_cuda(images)
        output = self.model(images)
        return output

    # Normal
    def forward_normal(self, input, target):
        images = input
        images = self.to_cuda(input)
        score  = self.model(images)
        loss = self.loss_func(score, target)
        if self.status == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return score, loss

    # Multihead
    def forward_multihead(self, input, target):
        target = target.view(target.shape[0], 1)
        images = self.to_cuda(input)
        if self.status == 'train':
            score, plain_scores, boosting_weights = self.model.forward(images, target)
            train_loss = self.loss_func(plain_scores, target, boosting_weights)
        else:
            score, _, _ = self.model.forward(images, target, calc_weight=False)
        loss = self.loss_func(score, target)
        if self.status == 'train':
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        return score, loss

    # Boosting
    def forward_boosting(self, input, target):
        target = target.view(target.shape[0], 1)
        images = self.to_cuda(input)
        if self.status == 'train':
            score, plain_scores, boosting_weights, aux_losses = self.model.forward(images, target)
            train_loss = self.loss_func(plain_scores, target, boosting_weights)
            # aux_losses[0]: activation  loss
            # aux_losses[1]: row-l2-norm loss
            train_loss = train_loss + 0.005 * aux_losses[0] + 0.05 * aux_losses[1]
        else:
            score, _, _, _ = self.model.forward(images, target, calc_weight=False)
        loss = self.loss_func(score, target)
        if self.status == 'train':
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        return score, loss

    def run_train_iteration(self, index, data, train_iters):
        self.status = 'train'
        self.call_hook_func('before_train_iteration_start')

        # Predict
        *inputs, targets = data
        target = torch.from_numpy(np.array(targets)).float().cuda(non_blocking=True)

        if 'multihead_siamese' in self.code:
            output, loss = self.forward_multihead(inputs, target)
        elif 'boosting_siamese' in self.code:
            output, loss = self.forward_boosting(inputs, target)
        else:
            output, loss = self.forward_normal(inputs, target)

        # Log
        for metric in self.metrics:
            self.metrics[metric]['train'].update(output, target)
        loss = loss.detach()
        logging.debug('[train {}/{}/{}] loss {}'
                     .format(self.current_epoch, index, train_iters, loss))
        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss

        self.call_hook_func('after_train_iteration_end')
        return loss

    def run_val_iteration(self, index, data, val_iters):
        self.status = 'val'
        self.call_hook_func('before_val_iteration_start')
        # Predict
        *inputs, targets = data
        target = torch.from_numpy(np.array(targets)).float().cuda(non_blocking=True)

        if 'multihead' in self.code:
            output, loss = self.forward_multihead(inputs, target)
        elif 'boosting' in self.code:
            output, loss = self.forward_boosting(inputs, target)
        else:
            output, loss = self.forward_normal(inputs, target)

        # Log
        for metric in self.metrics:
            self.metrics[metric]['val'].update(output, target)
        loss = loss.detach()
        logging.debug('[val {}/{}/{}] loss {}'.format(
            self.current_epoch, index, val_iters, loss))

        self.call_hook_func('after_val_iteration_end')
        return loss

    def persist(self, name):
        """save the model to disk
        """
        self.call_hook_func('before_checkpoint_persisted')
        if self.drawer is not None:
            drawer_state = self.drawer.get_state()
        else:
            drawer_state = {}

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'lowest_train_loss': self.lowest_train_loss,
            'lowest_val_loss': self.lowest_val_loss,
            'drawer_state': drawer_state,
            'cache': self.cache
        }

        torch.save(state, self.standard_model_path(name))
        logging.debug(f'save checkpoint to {self.standard_model_path(name)}')
        self.call_hook_func('after_checkpoint_persisted')

    def standard_model_path(self, model_name):
        return os.path.join(self.models_dir, f'{model_name}.pth.tar')

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

    def create_dirs(self):
        """Create directories
        """
        self.create_dir('')
        self.create_dir(self.code)
        self.create_dir(self.code, self.fold)
        self.create_dir(self.code, self.fold, 'models')

    def create_dir(self, *args):
        """Create directory
        """
        current_dir = self.alchemistic_directory
        for dir_name in args:
            current_dir = os.path.join(current_dir, dir_name)
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)

# ===============
class mean_aggregator():
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.sum = 0
        self.count = 0

    def update(self, target, output):
        self.sum += self.loss_func(target, output)
        self.count += len(target)

    def mean(self):
        _mean = self.sum / (self.count + 10e-15)
        if torch.is_tensor(_mean):
            return _mean.item()
        else:
            return _mean
