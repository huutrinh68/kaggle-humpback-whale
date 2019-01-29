import os
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

def get_multihot(targets, n_classes):
    labels = [np.array(list(map(int, str_label.split(' '))))
              for str_label in targets]
    y  = [np.eye(n_classes,dtype=np.float)[label].sum(axis=0)
          for label in labels]
    return y


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def imshow(img,text=None,should_save=False):
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(img, cmap='gray')
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
