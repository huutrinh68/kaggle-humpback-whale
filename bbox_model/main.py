import sys
import numpy as np
import pandas as pd

from keras import backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy.linalg import inv as mat_inv
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import BatchNormalization

from model import build_model
from transform import coord_transform, bounding_rectangle
from config import Config
from data import TrainingData, read_raw_image, read_for_validation, \
    get_train_data


old_stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
sys.stderr = old_stderr

config = Config()

def main():
    data = get_train_data()

    # Train data
    train, val = train_test_split(data, test_size=200, random_state=1)
    train += train
    train += train
    train += train
    len(train),len(val)

    # Valid data
    val_a = np.zeros((len(val),)+config.img_shape,
                     dtype=K.floatx()) # Preprocess validation images
    val_b = np.zeros((len(val),4),dtype=K.floatx()) # Preprocess bounding boxes
    for i,(p,coords) in enumerate(tqdm(val)):
        img,trans      = read_for_validation(p)
        coords         = coord_transform(coords, mat_inv(trans))
        x0,y0,x1,y1    = bounding_rectangle(coords, img.shape)
        val_a[i,:,:,:] = img
        val_b[i,0]     = x0
        val_b[i,1]     = y0
        val_b[i,2]     = x1
        val_b[i,3]     = y1

    # Train using cyclic learning rate
    for num in range(1, 4):
        model_name = 'cropping-%01d.h5' % num
    model = build_model()
    print(model_name)
    model.compile(Adam(lr=0.032), loss='mean_squared_error')
    model.fit_generator(
        TrainingData(train), epochs=50, max_queue_size=12, workers=4, verbose=1,
        validation_data=(val_a, val_b),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=9, min_delta=0.1, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.1, factor=0.25, min_lr=0.002, verbose=1),
            ModelCheckpoint(model_name, save_best_only=True, save_weights_only=True),
        ])
    model.load_weights(model_name)
    model.evaluate(val_a, val_b, verbose=0)

    # Now choose which model to use
    model.load_weights('cropping-1.h5')
    loss1 = model.evaluate(val_a, val_b, verbose=0)
    model.load_weights('cropping-2.h5')
    loss2 = model.evaluate(val_a, val_b, verbose=0)
    model.load_weights('cropping-3.h5')
    loss3 = model.evaluate(val_a, val_b, verbose=0)
    model_name = 'cropping-1.h5'
    if loss2 <= loss1 and loss2 < loss3: model_name = 'cropping-2.h5'
    if loss3 <= loss1 and loss3 <= loss2: model_name = 'cropping-3.h5'
    model.load_weights(model_name)

    # Variance normalization
    model2 = build_model(with_dropout=False)
    model2.load_weights(model_name)
    model2.compile(Adam(lr=0.002), loss='mean_squared_error')
    model2.evaluate(val_a, val_b, verbose=0)

    # Recompute the mean and variance running average without dropout
    for layer in model2.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    model2.compile(Adam(lr=0.002), loss='mean_squared_error')
    model2.fit_generator(TrainingData(), epochs=1, max_queue_size=12, workers=6, verbose=1, validation_data=(val_a, val_b))
    for layer in model2.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    model2.compile(Adam(lr=0.002), loss='mean_squared_error')
    model2.save('cropping.model')

    # Generate bounding boxes
    tagged = [p for _,p,_ in pd.read_csv(config.train_csv).to_records()]
    submit = [p for _,p,_ in pd.read_csv(config.sample_submission).to_records()]
    join = tagged + submit

    # If the picture is part of the bounding box dataset, use the golden value.
    p2bb = {}
    for i,(p,coords) in enumerate(data): p2bb[p] = bounding_rectangle(coords, read_raw_image(p).size)
    len(p2bb)

    # For other pictures, evaluate the model.
    p2bb = {}
    for p in tqdm(join):
        if p not in p2bb:
            img,trans         = read_for_validation(p)
            a                 = np.expand_dims(img, axis=0)
            x0, y0, x1, y1    = model2.predict(a).squeeze()
            (u0, v0),(u1, v1) = coord_transform([(x0,y0),(x1,y1)], trans)
            img               = read_raw_image(p)
            u0 = max(u0, 0)
            v0 = max(v0, 0)
            u1 = min(u1, img.size[0])
            v1 = min(v1, img.size[1])
            p2bb[p]           = (u0, v0, u1, v1)

    with open('cropping.txt', 'w') as f:
        for p in p2bb:
            u0, v0, u1, v1 = p2bb[p]
            f.write('{},{},{},{},{}\n'.format(str(p),
                                              str(u0),
                                              str(v0),
                                              str(u1),
                                              str(v1)))

if __name__ == '__main__':
    main()
