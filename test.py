# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import ast
import torch
from torch.backends import cudnn

import models
import DataSet
import json
from evaluations import Recall_at_ks, pairwise_similarity, extract_features
from utils.serialization import load_checkpoint, load_latest
from utils import display
cudnn.benchmark = True

def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, nThreads=16, batch_size=100, pool_feature=False, **kargs):
    dataset_name = data
    model = models.create(net, dim=dim, pretrained=False)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except:
        print(
            'load checkpoint failed, the state in the '
            'checkpoint is not matched with the model, '
            'try to reload checkpoint with unstrict mode')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(data, width=width, root=root)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True,
        num_workers=nThreads)
    test_loader = torch.utils.data.DataLoader(
        data.gallery, batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True,
        num_workers=nThreads)

    train_feature, train_labels \
        = extract_features(model, train_loader, print_freq=1e4,
                           metric=None, pool_feature=pool_feature)
    test_feature, test_labels \
        = extract_features(model, test_loader, print_freq=1e4,
                           metric=None, pool_feature=pool_feature)

    return train_feature, train_labels, test_feature, test_labels

def test(args):
    checkpoint = load_latest(args.resume)
    if checkpoint == None:
        print('{} is not avaible! Exit!'.format(args.resume))
        return

    epoch = checkpoint['epoch']
    train_feature, train_labels, test_feature, test_labels = \
        Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,
                       dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature)

    # train-train pairwise similarity
    sim_mat = pairwise_similarity(train_feature, train_feature)
    sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    recall_ks, ks = Recall_at_ks(sim_mat, query_ids=train_labels, gallery_ids=train_labels, data=args.data)
    result = '  '.join(['top@%d:%.4f' % (k, rc) for k, rc in zip(ks, recall_ks)])
    print('Epoch-%d' % epoch, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')

    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--resume', '-r', type=str, default='model.pkl', metavar='PATH')

    parser.add_argument('--dim', '-d', type=int, default=512,
                        help='Dimension of Embedding Feather')
    parser.add_argument('--width', type=int, default=224,
                        help='width of input image')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')
    parser.add_argument('--reload', type=str, default=None)
    parser.add_argument('--save_dir', default=None, help='where the trained models saved')

    args = parser.parse_args()
    if args.reload != None:
        config_path = 'logs/experiments/{}/config.json'.format(args.reload)
        with open(config_path) as f:
            config = json.load(f)['args']
            display(argparse.Namespace(**config))
            args.width    = config['width']
            args.dim      = config['dim']
            args.net      = config['net']
            args.resume   = config['save_dir']
            args.data     = config['data']
            args.data_root= config['data_root']

    test(args)
