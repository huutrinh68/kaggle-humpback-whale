class Config():
    def __init__(self):
        self.root = '/home/dizzyvn/workspace/kaggle/humpback-whale-identification/'
        self.sample_submission = self.root + 'input/sample_submission.csv'
        self.train_csv = self.root + 'input/train.csv'
        self.ouput_cropping = 'input/cropping.txt'
        self.data_paths = [
            self.root + 'input/cropping_data/train/',
            self.root + 'input/cropping_data/test/',
            self.root + 'input/keypoints_data/',
            self.root + 'input/train/',
            self.root + 'input/test/',
        ]
        self.cropping_annotation  = self.root + 'input/cropping_data/cropping.txt'
        self.keypoints_annotation = self.root + 'input/keypoints_data/keypoints.csv'
        self.img_shape  = (256,256,1)
        self.anisotropy = 2.15
        self.batch_size = 50
