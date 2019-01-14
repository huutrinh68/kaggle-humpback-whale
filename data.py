import pathlib
import gc
import cv2
import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from imgaug import augmenters as iaa
from collections import OrderedDict

# local import
from utils import get_multihot
from sacred import Ingredient
from path import path_ingredient
data_ingredient = Ingredient('data', ingredients=[path_ingredient])

iaa_dict = {'rot90' : iaa.Affine(rotate=90),
            'rot-90': iaa.Affine(rotate=-90),
            'rot180': iaa.Affine(rotate=180),
            'rot270': iaa.Affine(rotate=270),
            'shear' : iaa.Affine(shear=(-10, 10)),
            'flipud': iaa.Flipud(1),
            'fliplr': iaa.Fliplr(1),
            'noop'  : iaa.Noop()}


@data_ingredient.config
def cfg():
    image_size = 512            # image size
    n_channels = 4              # num of channels
    n_classes  = 28             # num of classes
    n_workers  = 4              # num of loader workers
    batch_size = 40             # batch size
    augment    = True           # train augment
    n_tta      = 3              # num of tta aug actions
    n_fold     = 5              # n fold
    upsampling = True
    aug_train  = ['noop', 'rot90', 'rot180', 'rot270', 'shear',
                  'flipud', 'fliplr']                             # train aug actions
    aug_tta    = ['noop', 'flipud', 'fliplr', 'rot90', 'rot-90']  # tta aug actions


@data_ingredient.capture
def load_gen(data_df, mode, path):
    if mode in ['train', 'val']:
        return CellDataset(data_df, path['root'] + path['train_data'], mode='train')
    if mode == 'test':
        return CellDataset(data_df, path['root'] + path['test_data'], mode='test')
    if mode == 'test-val':
        return CellDataset(data_df, path['root'] + path['train_data'], mode='test')


@data_ingredient.capture
def load_loader(dataset, mode, batch_size, n_workers):
    if mode == 'train':
        return DataLoader(dataset, batch_size = batch_size, shuffle=True,
                          pin_memory=True, num_workers=n_workers)
    if mode == 'val':
        return DataLoader(dataset, batch_size = batch_size, shuffle=False,
                          pin_memory=True, num_workers=n_workers)
    if mode in ['test', 'test-val']:
        return DataLoader(dataset, batch_size = 1, shuffle=False,
                          pin_memory=True, num_workers=n_workers)

@data_ingredient.capture
def create_loader(fold, n_fold, seed, debug, path, n_classes, upsampling):
    train_data_df, val_data_df = get_train_val_df(fold, n_fold, seed, debug)

    # Up sampling
    train_data_df = up_sampling(train_data_df, upsampling)

    # Create data loader
    train_gen = load_gen(data_df=train_data_df, mode='train')
    val_gen   = load_gen(data_df=val_data_df, mode='val')
    train_loader = load_loader(train_gen, 'train')
    val_loader   = load_loader(val_gen, 'val')
    labels_count = count_labels(train_data_df)

    return train_loader, val_loader, labels_count

@data_ingredient.capture
def create_test_loader(fold, n_fold, seed, debug, path):
    _, val_data_df = get_train_val_df(fold, n_fold, seed, debug)
    test_data_df = pd.read_csv(path['root'] + path['sample_submission'])
    if debug: test_data_df = test_data_df[:100]
    test_gen    = load_gen(data_df=test_data_df, mode='test')
    test_loader = load_loader(test_gen, mode='test')
    val_gen     = load_gen(data_df=val_data_df, mode='test-val')
    val_loader  = load_loader(val_gen, mode='test-val')

    return test_loader, val_loader

@data_ingredient.capture
def get_train_val_df(fold, n_fold, seed, debug, path, n_classes):
    # Load csv file
    kaggle_df   = pd.read_csv(path['root'] + path['kaggle_csv'])
    external_df = pd.read_csv(path['root'] + path['external_csv'])
    all_files   = pd.concat([kaggle_df, external_df])
    if debug: all_files = all_files[:100]

    # Get the fold-th split
    y = get_multihot(targets=all_files.Target, n_classes=n_classes)
    mskf = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for _fold, (train_index, val_index) in enumerate(mskf.split(all_files, y)):
        if _fold == fold: break # Get the fold-th split
    train_data_df = all_files.iloc[train_index].reset_index()
    val_data_df = all_files.iloc[val_index].reset_index()

    return train_data_df, val_data_df

@data_ingredient.capture
def up_sampling(train_data_df, upsampling):
    # Upsampling
    if upsampling:
        train_df_orig = train_data_df.copy()
        lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
        for i in lows:
            target = str(i)
            indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
            train_data_df = pd.concat([train_data_df, train_df_orig.loc[indicies]], ignore_index=True)
            indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target+" ")].index
            train_data_df = pd.concat([train_data_df, train_df_orig.loc[indicies]], ignore_index=True)
            indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" "+target)].index
            train_data_df = pd.concat([train_data_df, train_df_orig.loc[indicies]], ignore_index=True)
            indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" "+target+" ")].index
            train_data_df = pd.concat([train_data_df, train_df_orig.loc[indicies]], ignore_index=True)
        del train_df_orig
        gc.collect()

    return train_data_df


def count_labels(train_data_df):
    target = train_data_df.apply(lambda x: x['Target'].split(' '), axis=1)
    y = target.tolist()
    list_order = [str(i) for i in range(28)]
    mlb = MultiLabelBinarizer(list_order)
    y = mlb.fit_transform(y)
    labels_count = OrderedDict()
    count_classes = np.sum(y, axis=0)
    for i, count in enumerate(count_classes):
        labels_count[i] = count
    del target, y
    gc.collect()

    return labels_count

# ==============================
# dataset class
class CellDataset(Dataset):
    @data_ingredient.capture
    def __init__(self, images_df, data_path, mode, path, n_classes, n_channels,
                 image_size, augment, n_tta, aug_train, aug_tta):
        if not isinstance(data_path, pathlib.Path):
            base_path = pathlib.Path(data_path)
        self.images_df = images_df.copy()
        self.n_classes  = n_classes
        self.n_channels = n_channels
        self.image_size = image_size
        self.augment = augment
        self.mode  = mode
        self.n_tta = n_tta
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / x)
        self.mlb = MultiLabelBinarizer(classes = np.arange(0, n_classes))
        self.mlb.fit(np.arange(0, n_classes))
        self.aug_train = aug_train
        self.aug_tta = aug_tta

    def __len__(self):
        return len(self.images_df)

    def preprocess(self,X):
        return T.Compose([T.ToPILImage(),
                          T.ToTensor(),
                          T.Normalize(mean=[0.5,], std=[0.5,])])(X).float()

    def __getitem__(self,index):
        X = self.read_images(index)
        if 'Predicted' in self.images_df:
            labels = self.images_df.iloc[index].Predicted
        else:
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
        y  = np.eye(self.n_classes,dtype=np.float)[labels].sum(axis=0)

        if self.mode == 'train' or self.n_tta == None:
            if self.augment: X = self.train_augment(X)
            return self.preprocess(X) ,y
        else:
            Xs = [self.preprocess(aug_image)
                  for aug_image in self.tta_augment(X)]
            return Xs, y

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        #use only rgb channels
        if self.n_channels == 4:
            images = np.zeros(shape=(self.image_size, self.image_size, 4))
        else:
            images = np.zeros(shape=(self.image_size, self.image_size, 3))
        r = np.array(Image.open(filename+"_red.jpg"))
        g = np.array(Image.open(filename+"_green.jpg"))
        b = np.array(Image.open(filename+"_blue.jpg"))
        y = np.array(Image.open(filename+"_yellow.jpg"))
        images[:,:,0] = r.astype(np.uint8)
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = b.astype(np.uint8)
        if self.n_channels == 4:
            images[:,:,3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        return images

    @data_ingredient.capture
    def train_augment(self, image):
        aug = iaa.Sequential([iaa.OneOf([iaa_dict[action] for action in self.aug_train])],
                       random_order=True)
        image_aug = aug.augment_image(image)
        return image_aug

    @data_ingredient.capture
    def tta_augment(self, image):
        augs = [iaa_dict[action] for action in self.aug_tta]
        return [action.augment_image(image) for action in augs[:self.n_tta]]
