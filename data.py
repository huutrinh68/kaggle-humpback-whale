import cv2
import gc
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator

# local import
from utils import imshow
from sacred import Ingredient, Experiment
from path import path_ingredient
from metadata import meta_ingredient, create_p2size, create_p2h, \
    create_h2ps, create_h2p, create_h2ws, create_w2hs, \
    create_w2ts, read_raw_image, create_t2w

data_ingredient = Ingredient('data', ingredients=[path_ingredient,
                                                  meta_ingredient])
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
    image_size = 128            # image size
    n_workers  = 4              # num of loader workers
    batch_size = 40             # batch size
    augment    = True           # train augment
    n_tta      = 3              # num of tta aug actions
    anisotropy = 2.15
    crop_margin= 0.01
    upsampling = True
    same_ratio = 0.5
    valid_num  = 1000
    test_batch_size = 100
    n_pair     = 5
    # aug_train  = ['noop', 'rot90', 'rot180', 'rot270', 'shear',
    #               'flipud', 'fliplr']                             # train aug actions
    # aug_train  = ['noop', 'fliplr', 'shear']
    # aug_tta    = ['noop', 'flipud', 'fliplr', 'rot90', 'rot-90']  # tta aug actions;

# ============== FeatureDataset ==========
# ========================================
class FeatureDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)

@data_ingredient.capture
def create_feature_loader(train_features, test_features, n_workers, test_batch_size):
    train_feature_dataset = FeatureDataset(train_features)
    test_feature_dataset  = FeatureDataset(test_features)
    train_feature_loader  = DataLoader(train_feature_dataset, shuffle=False, pin_memory=True,
                                       num_workers=n_workers, batch_size=test_batch_size)
    test_feature_loader   = DataLoader(test_feature_dataset,  shuffle=False, pin_memory=True,
                                       num_workers=n_workers, batch_size=test_batch_size)
    return train_feature_loader, test_feature_loader

# ============== ImageDataset ==========
# ========================================
class ImageDataset(Dataset):
    @data_ingredient.capture
    def __init__(self, data_reader, mode,
                 batch_size, same_ratio, path, image_size):
        self.data_reader = data_reader
        self.same_ratio  = same_ratio
        self.batch_size  = batch_size
        self.image_size  = image_size
        self.mode        = mode

        if self.mode == 'train':
            self.p2h             = create_p2h(None)
            self.h2p             = create_h2p(None, None)
            self.t2w             = create_t2w(None)
            self.w2ts, self.data = create_w2ts(None)
        else:
            SUB_DF   = path['root'] + path['sample_submission']
            self.data = [p for _, p, _ in pd.read_csv(SUB_DF).to_records()]

    def __getitem__(self, index):
        tmp = self.data[index]
        tmp = self.data_reader.read_for_validation(tmp)
        tmp = self.preprocess(tmp)
        return tmp

    def __len__(self):
        return len(self.data)

    def preprocess(self,X):
       return T.Compose([T.ToTensor()])(X).float()


@data_ingredient.capture
def create_image_loader(batch_size, n_workers):
    data_reader     = cropDataGenerator()
    # TRAIN
    train_image_dataset = ImageDataset(data_reader, 'train')
    train_image_loader  = DataLoader(train_image_dataset, shuffle=False, pin_memory=True,
                                        num_workers=n_workers, batch_size=batch_size)
    # TEST
    test_image_dataset   = ImageDataset(data_reader, 'test')
    test_image_loader    = DataLoader(test_image_dataset, shuffle=False, pin_memory=True,
                                        num_workers=n_workers, batch_size=batch_size)
    return train_image_loader, test_image_loader

# ============== SiameseDataset ==========
# ========================================
class SiameseDataset(Dataset):
    @data_ingredient.capture
    def __init__(self, data_reader, mode, test_data, valid_num, model,
                 batch_size, same_ratio, path, image_size):
        self.data_reader = data_reader
        self.same_ratio  = same_ratio
        self.batch_size  = batch_size
        self.image_size  = image_size
        self.mode        = mode
        self.t2w         = create_t2w(None)
        self.w2ts, self.data = create_w2ts(None)
        self.t2i = {}
        for i, t in enumerate(self.data):
            self.t2i[t] = i
        self.model       = model

        # Create test set
        # these sample will not be used in training
        if test_data == None:
            tmp_data = []
            for i in range(valid_num):
                while True:
                    index = random.choice(range(len(self.data)))
                    whale = self.t2w[self.data[index]]
                    n_same_whale = len(self.w2ts[whale])
                    if n_same_whale > 10:
                        break
                # Positive (same whale)
                a = self.data[index]
                b = self.get_match(index)
                c = 1
                tmp_data.append((a,b,c))
                # Negative (different whale)
                a = self.data[index]
                b = self.get_unmatch(index)
                c = 0
                tmp_data.append((a,b,c))
            self.test_data = tmp_data
        else:
            self.test_data = test_data

        self.test_set = set(self.test_data)

    @data_ingredient.capture
    def __getitem__(self, index, n_pair):
        # Train mode
        if self.mode=='train':
            should_choose_same = random.uniform(0, 1)
            a = self.data[index]
            _pos_b = []
            # Generate match pair
            for i in range(n_pair):
                while True:
                    b = self.get_match(index)
                    if (a,b,1) not in self.test_set and \
                       (b,a,1) not in self.test_set: break
                _pos_b.append(b)
            _a_img  = self.preprocess(self.data_reader.read_for_training(a))
            _a_tensor = _a_img.view(1,1,self.image_size, self.image_size)
            _a_feature = self.model.module.get_features(_a_tensor.cuda())

            _pos_b_imgs = [self.preprocess(self.data_reader.read_for_training(b))
                           for b in _pos_b]
            _pos_b_tensor = torch.cat(_pos_b_imgs, 0).reshape(n_pair, 1,
                                                              self.image_size,
                                                              self.image_size)

            # Generate unmatch pair
            _neg_b = []
            n_trial = 2*n_pair
            for i in range(n_trial):
                while True:
                    b = self.get_unmatch(index)
                    if (a,b,0) not in self.test_set and \
                       (b,a,0) not in self.test_set: break
                _neg_b.append(b)

            # Calculate score for negative samples
            _neg_b_imgs = [self.preprocess(self.data_reader.read_for_training(b))
                           for b in _neg_b]
            _neg_b_tensor = torch.cat(_neg_b_imgs, 0).reshape(n_trial, 1,
                                                              self.image_size,
                                                              self.image_size)
            _neg_b_feature = self.model.module.get_features(_neg_b_tensor.cuda())

            # Manipulate a_feature to get the same shape with b
            _a_feature = _a_feature.repeat((_neg_b_feature.shape[0], 1))

            # Calculate score
            scores = self.model.module.get_score(_a_feature, _neg_b_feature)\
                                      .detach().cpu().numpy()
            hard_index = np.argsort(scores)[-n_pair:].tolist()

            # Get tensor
            _a_tensor = _a_tensor.view(self.image_size, self.image_size)
            _a_tensor = _a_tensor.repeat(n_pair, 1, 1, 1)
            _neg_b_tensor = _neg_b_tensor[hard_index, :, :, :]

            _a = torch.cat((_a_tensor, _a_tensor), 0)
            _b = torch.cat((_pos_b_tensor, _neg_b_tensor), 0)
            _c = torch.tensor([1] * n_pair + [0] * n_pair)

        # Test mode
        else:
            a,b,c = self.test_data[index]
            a = self.data_reader.read_for_training(a)
            b = self.data_reader.read_for_training(b)
            _a = self.preprocess(a)
            _b = self.preprocess(b)
            _c = c

        return _a, _b, _c

    def get_match(self, idx):
        whale = self.t2w[self.data[idx]]
        t = random.choice(self.w2ts[whale])
        return t

    def get_unmatch(self, idx):
        whale = self.t2w[self.data[idx]]
        while True:
            t = random.choice(self.data)
            if whale != self.t2w[t]:
                return t

    def set_seed(self, seed):
        random.seed(seed)

    @data_ingredient.capture
    def __len__(self, valid_num):
        if self.mode == 'train':
            return len(list(self.data))
        else:
            return len(list(self.test_data))

    def preprocess(self,X):
       return T.Compose([T.ToTensor()])(X).float()


@data_ingredient.capture
def create_siamese_loader(model, batch_size, n_workers):
    data_reader     = cropDataGenerator()
    # VALID
    val_siamese_dataset   = SiameseDataset(data_reader, 'test', test_data=None, model=model)
    val_siamese_loader    = DataLoader(val_siamese_dataset, shuffle=False, pin_memory=True,
                                       # num_workers = n_workers,
                                       batch_size=batch_size)

    # TRAIN
    test_data = val_siamese_dataset.test_data
    train_siamese_dataset = SiameseDataset(data_reader, 'train', test_data=test_data, model=model)
    train_siamese_loader  = DataLoader(train_siamese_dataset, shuffle=True, pin_memory=True,
                                       # num_workers = n_workers,
                                       batch_size=batch_size)

    return train_siamese_loader, val_siamese_loader


# ============== cropDataGenerator ========
# =========================================
class cropDataGenerator():
    @data_ingredient.capture
    def __init__(self, path, crop_margin, anisotropy, image_size):
        TRAIN_DF = path['root'] + path['train_csv']
        SUB_DF   = path['root'] + path['sample_submission']
        BB_DF    = path['root'] + path['bbox']
        self.tagged = dict([(p, w) for _, p, w in pd.read_csv(TRAIN_DF).to_records()])
        self.submit = [p for _, p, _ in pd.read_csv(SUB_DF).to_records()]
        self.join   = list(self.tagged.keys()) + self.submit
        self.p2h    = create_p2h(self.join)
        self.p2bb   = pd.read_csv(BB_DF).set_index("Image")
        self.crop_margin = crop_margin
        self.anisotropy  = anisotropy
        self.image_size  = image_size

        self.h2p              = create_h2p(None, None)
        self.w2ts, self.train = create_w2ts(None)
        self.datagen          = ImageDataGenerator(rotation_range=2,
                                                    width_shift_range=1,
                                                    height_shift_range=1,
                                                    shear_range=2,
                                                    horizontal_flip=True,
                                                    vertical_flip=False,
                                                    samplewise_std_normalization=True,
                                                    fill_mode='nearest')

    @data_ingredient.capture
    def read_cropped_image(self, p, augment=True):
        """
        @param p : the name of the picture to read
        @param augment: True/False if data augmentation should be performed
        @return a numpy array with the transformed image
        """
        # If an image id was given, convert to filename
        if p in self.h2p:
            p = self.h2p[p]
        img = read_raw_image(p).convert('L')
        img = np.asarray(img)
        row = self.p2bb.loc[p]
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']

        # Fix the bounding box
        size_y, size_x = img.shape
        dx = x1 - x0
        dy = y1 - y0
        x0 -= dx * self.crop_margin
        x1 += dx * self.crop_margin + 1
        y0 -= dy * self.crop_margin
        y1 += dy * self.crop_margin + 1

        # Crop
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > size_x: x1 = size_x
        if y1 > size_y: y1 = size_y

        if y1 > y0 and x1 > x0:
            img = img[int(y0):int(y1), int(x0):int(x1)]

        # Resize and normalize
        y, x = img.shape
        if x > y: ratio = self.image_size / x
        else:     ratio = self.image_size / y
        if int(x * ratio) != 0 and int(y * ratio) != 0:
            # img = cv2.resize(img, (int(x * ratio), int(y * ratio))).astype(float)
            img = cv2.resize(img, (self.image_size, self.image_size)).astype(float)
            img -= np.mean(img, keepdims=True)
            img += np.abs(np.min(img))
            img /= np.std(img, keepdims=True)
            img  = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

        # Padding
        y, x = img.shape
        new_img = np.ones((self.image_size, self.image_size)) * 230.0
        _x = int((self.image_size - x)/2)
        _y = int((self.image_size - y)/2)
        new_img[_y:_y+y,_x:_x+x] = img
        new_img = new_img.reshape(self.image_size, self.image_size, 1)
        # Agumentation
        if augment:
            new_img = self.train_augment(new_img)
        return new_img.reshape(self.image_size, self.image_size, 1)


    def read_for_training(self, p):
        return self.read_cropped_image(p, augment=True)


    def read_for_validation(self, p):
        return self.read_cropped_image(p, augment=False)


    def train_augment(self, image):
        image = image.reshape((1,) + image.shape)
        image_aug = self.datagen.flow(image, batch_size=1)[0]
        return image_aug

# ============ TEST ===========
ex = Experiment('tmp', ingredients=[data_ingredient,
                                    meta_ingredient])

@ex.main
def main(data, path):
    train_loader, test_loader =  create_siamese_loader()
    data_reader     = cropDataGenerator()
    siamese_dataset = SiameseDataset(data_reader, 'train')
    visual_dataloader = DataLoader(siamese_dataset,
                                   shuffle=True,
                                   num_workers=4,
                                   batch_size=16)
    image_size = data['image_size']
    data_iter = iter(visual_dataloader)
    example_batch = next(data_iter)
    for i in range(8):
        a = example_batch[0][i,:,:]
        b = example_batch[1][i,:,:]
        a = a.reshape((image_size, image_size))
        b = b.reshape((image_size, image_size))
        print(example_batch[2][i])
        # imshow(a)
        # imshow(b)

if __name__ == '__main__':
    ex.run_commandline()
