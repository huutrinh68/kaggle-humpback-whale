import random
import numpy as np

from PIL import Image as pil_image
from os.path import isfile
from keras.utils import Sequence
from numpy.linalg import inv as mat_inv
from keras.preprocessing.image import img_to_array
from keras import backend as K

from config import Config
from transform import coord_transform, center_transform, bounding_rectangle, \
    transform_img, build_transform

config = Config()

class TrainingData(Sequence):
    def __init__(self, train_data):
        super(TrainingData, self).__init__()
        self.batch_size = config.batch_size
        self.data = train_data
    def __getitem__(self, index):
        start = self.batch_size*index;
        end   = min(len(self.data), start + self.batch_size)
        size  = end - start
        a     = np.zeros((size,) + config.img_shape, dtype=K.floatx())
        b     = np.zeros((size,4), dtype=K.floatx())
        for i,(p,coords) in enumerate(self.data[start:end]):
            img,trans   = read_for_training(p)
            coords      = coord_transform(coords, mat_inv(trans))
            x0,y0,x1,y1 = bounding_rectangle(coords, img.shape)
            a[i,:,:,:]  = img
            b[i,0]      = x0
            b[i,1]      = y0
            b[i,2]      = x1
            b[i,3]      = y1
        return a,b
    def __len__(self):
        return (len(self.data) + self.batch_size - 1)//self.batch_size

# Read an image as black&white numpy array
def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)

def read_raw_image(p):
    return pil_image.open(expand_path(p))

def expand_path(p):
    for path in config.data_paths:
        if isfile(path + p): return path + p
    return p

def get_train_data():
    with open(config.cropping_annotation, 'rt') as f: data = f.read().split('\n')[:-1]
    data   = [line.split(',') for line in data]
    data_1 = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]

    with open(config.keypoints_annotation, 'r') as f:data = f.read().strip().split('\n')[:-1]
    data   = [line.split(',') for line in data]
    data_2 = []
    for p, *coord in data:
        tmp = []
        for i in range(0, len(coord),2):
            try:
                x = round(float(coord[i]))
                y = round(float(coord[i+1]))
                tmp.append((x, y))
            except:
                pass
        data_2.append((p, tmp))
    data = data_1 + data_2
    return data

# Read an image for validation, i.e. without data augmentation.
def read_for_validation(p):
    x  = read_array(p)
    t  = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t

# Read an image for training, i.e. including a random affine transformation
def read_for_training(p):
    x  = read_array(p)
    t  = build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.9, 1.0),
            random.uniform(0.9, 1.0),
            random.uniform(-0.05*config.img_shape[0], 0.05*config.img_shape[0]),
            random.uniform(-0.05*config.img_shape[1], 0.05*config.img_shape[1]))
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t

if __name__ == '__main__':
    pass
