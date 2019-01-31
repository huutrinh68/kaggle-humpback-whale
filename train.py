import random
import torch
import warnings
import os
import numpy as np
import pandas as pd
from skeleton.trainer import Trainer
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
from data import data_ingredient, create_siamese_loader
from path import path_ingredient, prepair_dir

ex = Experiment('Train', ingredients=[model_ingredient,      # model
                                      optimizer_ingredient,  # optimizer
                                      data_ingredient,       # data
                                      path_ingredient,       # path
                                      criterion_ingredient]) # criterion

# ex.observers.append(MongoObserver.create(db_name='humpback_whale'))
# ex.observers.append(FileStorageObserver.create('exp_logs/experiments'))
# ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    max_epochs = 20
    threshold = 0.2
    resume = True
    debug = False
    comment = ''
    seed=2050
    if debug == True:
        max_epochs = 3

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
def main(_log, max_epochs, resume, model, optimizer, data, path, seed, threshold, debug, criterion):
    # Model
    device = init()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(load_model()).to(device)
    else:
        model = load_model().to(device)

    # Optimizer
    optimizer = load_optimizer(model.parameters())

    # Data
    train_siamese_loader, val_siamese_loader = create_siamese_loader()

    # Loss function
    loss_func = load_loss()

    # Training
    # iteration_number= 0
    # loss_history = []
    # counter = []
    # for epoch in range(0, max_epochs):
    #     for i, data in tqdm(enumerate(siamese_loader)):
    #         img0, img1, label = data
    #         img0, img1, label = img0.cuda().float(), img1.cuda().float(), label.cuda().float()
    #         optimizer.zero_grad()
    #         output1, output2 = model(img0, img1)
    #         loss_contrastive = loss_func((output1, output2), label)
    #         loss_contrastive.backward()
    #         optimizer.step()
    #         if i % 10 == 0 :
    #             print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
    #             iteration_number +=10
    #             counter.append(iteration_number)
    #             loss_history.append(loss_contrastive.item())
    # show_plot(counter,loss_history)

    trainer = Trainer(
        alchemistic_directory = path['root'] + path['exp_logs'] + 'checkpoints/',
        code = get_dir_name(),
        model=model,
        optimizer=optimizer,
        train_dataloader=train_siamese_loader,
        val_dataloader=val_siamese_loader,
        loss_func=loss_func,
        max_epochs=max_epochs,
        resume=resume,
        hooks = {'after_load_checkpoint': after_load_checkpoint,
                 'after_epoch_end': after_epoch_end},
    )

    if debug:
        trainer.train()
    else:
        try:
            trainer.train()
        except Exception as e:
            _log.error('Unexpected exception! %s', e)

@ex.capture
def after_epoch_end(trainer, _run):
    pass


@ex.capture
def after_load_checkpoint(trainer, _run):
    pass


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
