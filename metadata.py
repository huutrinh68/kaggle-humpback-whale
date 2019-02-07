import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import sqrt
from PIL import Image
from imagehash import phash
from tqdm import tqdm
from os.path import isfile

from sacred import Ingredient, Experiment
from path import path_ingredient
from utils import save_pickle, load_pickle

meta_ingredient = Ingredient('meta', ingredients=[path_ingredient])
ex = Experiment('tmp', ingredients=[meta_ingredient])

def match(h1, h2, h2ps):
    '''
    Function to check if hash-1 is match with hash-2
    '''
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = Image.open(expand_path(p1))
            i2 = Image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True

@meta_ingredient.capture
def expand_path(p, path):
    TRAIN = path['root'] + path['train_data']
    TEST  = path['root'] + path['test_data']
    if isfile(TRAIN + p): return TRAIN + p
    if isfile(TEST  + p): return TEST + p
    return None

@meta_ingredient.capture
def read_raw_image(p):
    img = Image.open(expand_path(p))
    return img

@meta_ingredient.capture
def create_p2size(join, path):
    '''
    [path -> size]
    Create a dictionary that map an image path
    with the size of that image
    '''
    p2size_path = path['root'] + path['p2size']
    if isfile(p2size_path):
        print(p2size_path, 'exists! Load it!')
        p2size = load_pickle(p2size_path)
    else:
        p2size = {}
        for p in tqdm(join):
            size = Image.open(expand_path(p)).size
            p2size[p] = size

        # Save to pickle
        save_pickle(p2size, p2size_path)

    return p2size

@meta_ingredient.capture
def create_p2h(join, path):
    '''
    [path -> size]
    Create a dictionary that map an image path with the hash of that image
    using phash. After that we find images which are close to each other
    and assign a same hash for these images.

    '''
    p2h_path = path['root'] + path['p2h']
    if isfile(p2h_path):
        print(p2h_path, 'exists! Load it!')
        p2h = load_pickle(p2h_path)
    else:
        # Compute phash for each image in the training and test set.
        p2h = {}
        for p in tqdm(join):
            img = Image.open(expand_path(p))
            h = phash(img)
            p2h[p] = h

        # Find all images associated with a given phash value.
        h2ps = {}
        for p, h in p2h.items():
            if h not in h2ps: h2ps[h] = []
            if p not in h2ps[h]: h2ps[h].append(p)

        # Find all distinct phash values
        hs = list(h2ps.keys())

        # If the images are close enough, associate the two phash values
        h2h = {}
        for i, h1 in enumerate(tqdm(hs)):
            for h2 in hs[:i]:
                if h1 - h2 <= 6 and match(h1, h2, h2ps):
                    s1 = str(h1)
                    s2 = str(h2)
                    if s1 < s2: s1, s2 = s2, s1
                    h2h[s1] = s2

        # Group together images with equivalent phash, and replace by string
        # format of phash (faster and more readable)
        for p, h in p2h.items():
            h = str(h)
            if h in h2h: h = h2h[h]
            p2h[p] = h

        # Save to pickle
        save_pickle(p2h, p2h_path)

    return p2h

@meta_ingredient.capture
def create_h2ps(p2h, path):
    '''
    Create a dictionary that map a hash to image paths associated with
    that hash value. Note that one hash value can maps to multiple image paths.
    '''
    h2ps_path = path['root'] + path['h2ps']
    if isfile(h2ps_path):
        print(h2ps_path, 'exists! Load it!')
        h2ps = load_pickle(h2ps_path)
    else:
        h2ps = {}
        for p, h in p2h.items():
            if h not in h2ps: h2ps[h] = []
            if p not in h2ps[h]: h2ps[h].append(p)
        save_pickle(h2ps, h2ps_path)
    return h2ps

def prefer(ps, p2size):
    '''
    Select prefer image in image paths 'ps'
    A image is defined to be better if it has higher resolution
    '''
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        # Select the image with highest resolution
        if s[0] * s[1] > best_s[0] * best_s[1]:
            best_p = p
            best_s = s
    return best_p

@meta_ingredient.capture
def create_h2p(h2ps, p2size, path):
    '''
    Assign a single best image when a hash is mapped with
    multiple image paths. Using prefer()
    '''
    h2p_path = path['root'] + path['h2p_prefer']
    if isfile(h2p_path):
        print(h2p_path, 'exists! Load it!')
        h2p = load_pickle(h2p_path)
    else:
        h2p = {}
        for h, ps in h2ps.items():
            h2p[h] = prefer(ps, p2size)
        save_pickle(h2p, h2p_path)
    return h2p

@meta_ingredient.capture
def create_h2ws(p2h, tagged, path):
    '''
    Create a dictionary that map a hash value to whale ids.
    Note that a hash can associated with multiple whale ids.
    We will use this to filter noisy samples.
    '''
    h2ws_path = path['root'] + path['h2ws']
    if isfile(h2ws_path):
        print(h2ws_path, 'exists! Load it!')
        h2ws = load_pickle(h2ws_path)
    else:
        h2ws = {}
        new_whale = 'new_whale'
        for p, w in tagged.items():
            # Use only identified whales
            if w != new_whale:
                h = p2h[p]
                if h not in h2ws: h2ws[h] = []
                if w not in h2ws[h]: h2ws[h].append(w)
        for h, ws in h2ws.items():
            if len(ws) > 1:
                h2ws[h] = sorted(ws)
        save_pickle(h2ws, h2ws_path)
    return h2ws

@meta_ingredient.capture
def create_w2hs(h2ws, path):
    '''
    Now we create a inverse dictionary to map a whale id with corresponding hash
    values of that whale. We only choose hash value that has a unique whale label.
    '''
    w2hs_path = path['root'] + path['w2hs']
    if isfile(w2hs_path):
        print(w2hs_path, 'exists! Load it!')
        w2hs = load_pickle(w2hs_path)
    else:
        w2hs = {}
        for h, ws in h2ws.items():
            # Use only unambiguous pictures
            if len(ws) == 1:
                w = ws[0]
                if w not in w2hs: w2hs[w] = []
                if h not in w2hs[w]: w2hs[w].append(h)
        for w, hs in w2hs.items():
            if len(hs) > 1:
                w2hs[w] = sorted(hs)
        save_pickle(w2hs, w2hs_path)
    return w2hs

@meta_ingredient.capture
def create_w2ts(w2hs, path):
    '''
    Create a dictionary that map a whale id with the training samples
    '''
    w2ts_path = path['root'] + path['w2ts']
    train_ps_path = path['root'] + path['train_ps']
    if isfile(w2ts_path):
        print(w2ts_path, 'exists! Load it!')
        w2ts = load_pickle(w2ts_path)
        train = load_pickle(train_ps_path)
    else:
        train = []  # A list of training image ids
        for hs in w2hs.values():
            if len(hs) > 1:
                train += hs
        random.shuffle(train)
        train_set = set(train)

        w2ts = {}  # Associate the image ids from train to each whale id.
        for w, hs in w2hs.items():
            for h in hs:
                if h in train_set:
                    if w not in w2ts:
                        w2ts[w] = []
                    if h not in w2ts[w]:
                        w2ts[w].append(h)
        for w, ts in w2ts.items():
            w2ts[w] = np.array(ts)
        save_pickle(w2ts, w2ts_path)
        save_pickle(train, train_ps_path)
    return w2ts, train

@meta_ingredient.capture
def create_t2w(w2ts, path):
    '''
    Create a dictionary that map a train data with the whale
    '''
    t2w_path = path['root'] + path['t2w']
    train_ps_path = path['root'] + path['train_ps']
    if isfile(t2w_path):
        print(t2w_path, 'exists! Load it!')
        t2w = load_pickle(t2w_path)
    else:
        t2w = {}
        for whale in w2ts:
            for t in w2ts[whale]:
                t2w[t] = whale
        save_pickle(t2w, t2w_path)
    return t2w

@ex.automain
def main(path):
    TRAIN_DF = path['root'] + path['train_csv']
    SUB_DF   = path['root'] + path['sample_submission']
    tagged = dict([(p, w) for _, p, w in pd.read_csv(TRAIN_DF).to_records()])
    submit = [p for _, p, _ in pd.read_csv(SUB_DF).to_records()]
    join = list(tagged.keys()) + submit

    p2size = create_p2size(join)
    p2h    = create_p2h(join)
    h2ps   = create_h2ps(p2h)
    h2p    = create_h2p(h2ps, p2size)
    h2ws   = create_h2ws(p2h, tagged)
    w2hs   = create_w2hs(h2ws)
    w2ts, train   = create_w2ts(w2hs)
    t2w    = create_t2w(w2ts)
