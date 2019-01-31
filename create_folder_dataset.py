import os
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

root_path = "/home/tran/workspace/humpback-whale-identification/data/"
folder_dataset = root_path + 'folder_dataset/'

df = pd.read_csv(root_path + 'train.csv')

# create train folder_dataset
for i in tqdm(range(len(df))):
    # get img_id
    folder_name = df.iloc[i].Id + '/'

    # get img_name
    img_name = df.iloc[i].Image

    # create folder_dataset
    create_dir(folder_dataset + folder_name)

    # copy original img to folder_dataset
    src = root_path + 'train/' + img_name
    dst = folder_dataset + folder_name + img_name
    copyfile(src, dst)
