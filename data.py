import cv2
import gc
import random
import numpy as np
import pandas as pd

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
    image_size = 512            # image size
    n_workers  = 4              # num of loader workers
    batch_size = 40             # batch size
    augment    = True           # train augment
    n_tta      = 3              # num of tta aug actions
    anisotropy = 2.15
    crop_margin= 0.01
    upsampling = True
    same_ratio = 0.5
    valid_num  = 1000
    # aug_train  = ['noop', 'rot90', 'rot180', 'rot270', 'shear',
    #               'flipud', 'fliplr']                             # train aug actions
    # aug_train  = ['noop', 'fliplr', 'shear']
    # aug_tta    = ['noop', 'flipud', 'fliplr', 'rot90', 'rot-90']  # tta aug actions;

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
            img = cv2.resize(img, (int(x * ratio), int(y * ratio))).astype(float)
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


# ============== siameseDataset ==========
# ========================================
class SiameseDataset(Dataset):
    @data_ingredient.capture
    def __init__(self, data_reader, mode, trained_pairs,
                 batch_size, same_ratio, path, image_size):
        self.data_reader = data_reader
        self.same_ratio  = same_ratio
        self.batch_size  = batch_size
        self.image_size  = image_size
        self.mode        = mode
        self.t2w         = create_t2w(None)
        self.w2ts, self.train = create_w2ts(None)
        self.t2i = {}
        for i, t in enumerate(self.train):
            self.t2i[t] = i
        self.trained_pairs = trained_pairs

    def __getitem__(self, index):
        if self.mode=='train':
            should_choose_same = random.uniform(0, 1)
            if should_choose_same < self.same_ratio:
                a = self.train[index]
                b = self.get_match(index)
                c = 0
            else:
                a = self.train[index]
                b = self.get_unmatch(index)
                c = 1

            # Adding index of a, b into trained_pairs for validation purpose
            ia = self.t2i[a]
            ib = self.t2i[b]
            if ia > ib: tmp=ia; ia=ib; ib=tmp
            self.trained_pairs.add((ia, ib))
            a = self.data_reader.read_for_training(a)
            b = self.data_reader.read_for_training(b)

        else:
            index = random.choice(range(len(self.train)))
            should_choose_same = random.uniform(0, 1)
            if should_choose_same < self.same_ratio:
                a = self.train[index]
                b = self.get_match(index)
                c = 0
            else:
                a = self.train[index]
                b = self.get_unmatch(index, filter=self.trained_pairs)
                c = 1
            a = self.data_reader.read_for_validation(a)
            b = self.data_reader.read_for_validation(b)

        _a = self.preprocess(a)
        _b = self.preprocess(b)
        return _a, _b, c

    def get_match(self, idx):
        whale = self.t2w[self.train[idx]]
        t = random.choice(self.w2ts[whale])
        return t

    def get_unmatch(self, idx, filter=set()):
        whale = self.t2w[self.train[idx]]
        while True:
            t = random.choice(self.train)
            if whale != self.t2w[t] and t not in filter:
                return t

    @data_ingredient.capture
    def __len__(self, valid_num):
        if self.mode == 'train':
            return len(list(self.train))
        else:
            return valid_num

    def preprocess(self,X):
       return T.Compose([T.ToTensor()])(X).float()


@data_ingredient.capture
def create_siamese_loader(trained_pairs, batch_size, n_workers):
    data_reader     = cropDataGenerator()
    # TRAIN
    train_siamese_dataset = SiameseDataset(data_reader, 'train', trained_pairs)
    train_siamese_loader  = DataLoader(train_siamese_dataset, shuffle=True, pin_memory=True,
                                        num_workers=n_workers, batch_size=batch_size)
    # VALID
    val_siamese_dataset   = SiameseDataset(data_reader, 'test', trained_pairs)
    val_siamese_loader    = DataLoader(val_siamese_dataset, shuffle=False, pin_memory=True,
                                        num_workers=n_workers, batch_size=batch_size)
    return train_siamese_loader, val_siamese_loader

# ============ TEST ===========
# ex = Experiment('tmp', ingredients=[data_ingredient,
#                                     meta_ingredient])

# @ex.main
# def main(data, path):
#     train_loader, test_loader =  create_siamese_loader()
#     print(len(train_loader.dataset.trained_pairs))
#     print(len(test_loader.dataset.trained_pairs))
#     data_reader     = cropDataGenerator()
#     siamese_dataset = SiameseDataset(data_reader, 'train')
#     visual_dataloader = DataLoader(siamese_dataset,
#                                    shuffle=True,
#                                    num_workers=4,
#                                    batch_size=16)
#     image_size = data['image_size']
#     data_iter = iter(visual_dataloader)
#     example_batch = next(data_iter)
#     for i in range(8):
#         a = example_batch[0][i,:,:]
#         b = example_batch[1][i,:,:]
#         a = a.reshape((image_size, image_size))
#         b = b.reshape((image_size, image_size))
#         print(example_batch[2][i])
#         imshow(a)
#         imshow(b)

# if __name__ == '__main__':
#     ex.run_commandline()

