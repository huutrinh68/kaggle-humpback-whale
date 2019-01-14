import os
import pandas as pd
from PIL import Image

root_path = "/home/trinhnh1/Documents/kaggle/human-protein/input/"
upsamplingpath = "/home/trinhnh1/Documents/kaggle/human-protein/input/upsampling"

def load_train_data():
    # minority set inside external data
    path1 = root_path + "Ex_minority.csv"

    # minority set inside kaggle raw data
    path2 = root_path + "Rw_minority.csv"

    # kaggle raw data
    path3 = root_path + "train.csv"
    
    # external data
    path4 = root_path + "external_data/img/train.csv"


    # load data
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)

    df = pd.concat([df3, df4, df1, df2])

    return df

if __name__ == '__main__':
    if not os.path.exists(upsamplingpath):
        os.makedirs(upsamplingpath)

    df = load_train_data()
    image = df.iloc[0]
    image_id = image[0]
    image_target = image[1]

    # Need to rewrite
    # image_path = root_path + "image_path" + ".jpg"

    # input = Image.open(image_path)

    # Augmentation
    #output = .....

    # Create new information of augmented image
    upsampling_csv = root_path + "upsampling.csv"
    #new_information = ["new iamge id", "copy target"]
    new_data = []
    new_information = ['fsfsdf', '0,15']
    new_data.append(new_information)
    upsample_data = pd.DataFrame(data=new_data, columns=['Id','Target'])
    print(upsample_data.shape)
    upsample_data.to_csv(upsampling_csv, index=False)

    # Save output
    # output.save("path_to_save_img")

    print("image_id: " + image_id)
    print("image_target: " + image_target)




