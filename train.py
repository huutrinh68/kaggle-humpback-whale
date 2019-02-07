# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import json
import torch.utils.data

from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint, load_latest
from trainer import train
from utils import orth_reg

import DataSet
import numpy as np
import os.path as osp

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.commands import print_config

ex = Experiment('Train')
ex.observers.append(MongoObserver.create(db_name='humpback_whale'))
ex.observers.append(FileStorageObserver.create('logs/experiments'))

cudnn.benchmark = True
use_gpu = True


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

@ex.config
def cfg():
    args = None

@ex.main
def main(_run):
    # s_ = time.time()
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    model = models.create(args.net, pretrained=True, dim=args.dim)

    # for vgg and densenet
    if args.resume in ['False', False, None]:
        weight = model.state_dict()
    else:
        # resume model
        if args.resume in ['latest', True]:
            chk_pt = load_latest(args.save_dir)
        else:
            chk_pt = load_checkpoint(args.resume)
        if chk_pt != None:
            weight = chk_pt['state_dict']
            start = chk_pt['epoch']
            model.load_state_dict(weight)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40*'#', 'BatchNorm NOT frozen')

    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.module._classifier.parameters()))

    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.module.parameters() if
                   id(p) not in new_param_ids]

    param_groups = [
                {'params': base_params, 'lr_mult': 0.0},
                {'params': new_params, 'lr_mult': 1.0}]

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
    criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha).cuda()

    # Decor_loss = losses.create('decor').cuda()
    data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    # save the train information

    for epoch in range(start, args.epochs):

        train(epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args,
              run=_run)

        if epoch == 1:
            optimizer.param_groups[0]['lr_mul'] = 0.1

        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=True, type=bool, required=False, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='whale', required=False,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--net', default='resnet18')
    parser.add_argument('--loss', default='batchall', required=False,
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    parser.add_argument('--resume', '-r', default=True,
                        help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)
    parser.add_argument('--reload', type=str, default=None)

    args = parser.parse_args()
    if args.save_dir == None:
        args.save_dir \
            = 'logs/checkpoints/' + \
            'net_{}_{}_'.format(args.net, args.dim) + \
            'loss_{}_{}_{}_{}_'.format(args.loss, args.alpha, args.beta, args.margin) + \
            'data_{}_{}_{}'.format(args.width, args.batch_size, args.num_instances)

    if args.reload != None:
        resume = args.resume
        config_path = 'logs/experiments/{}/config.json'.format(args.reload)
        with open(config_path) as f:
            config = json.load(f)['args']
            config['resume'] = resume
            args = argparse.Namespace(**config)

    ex.run(config_updates={'args':args})
