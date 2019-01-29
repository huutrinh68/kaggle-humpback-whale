## README
This is a refactored version of the kernel [Bounding Box Model(martinpiotte)](https://www.kaggle.com/martinpiotte/bounding-box-model/data) which is modified so that it can use the new annotation data from [Humpback Whale Fluke keypoints(Paul Johnson)](https://www.kaggle.com/oewyn000/humpback-whale-fluke-keypoints).

### How to use
1. Download train, test and cropping annotation data from [Bounding Box Model(martinpiotte)](https://www.kaggle.com/martinpiotte/bounding-box-model/data) and extract into `$ROOT/input/cropping_data/train/`, `$ROOT/input/cropping_data/test/`, `$ROOT/input/cropping_data/cropping.txt` respectively
2. Download images and keypoints annotation data from [Humpback Whale Fluke keypoints(Paul Johnson)](https://www.kaggle.com/oewyn000/humpback-whale-fluke-keypoints) and extract into `$ROOT/input/keypoints_data/`
3. Run main.py to train and create bbox prediction, results is saved into `$ROOT/input/cropping.txt`
