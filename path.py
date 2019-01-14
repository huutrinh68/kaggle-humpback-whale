from utils import create_dir
# sacred import
from sacred import Ingredient
path_ingredient = Ingredient('path')


@path_ingredient.config
def cfg():
    root              = '/home/tran/workspace/new_human_protein_atlas_image_classfication/'
    exp_logs          = 'exp_logs/'
    submit            = 'submit/'
    train_data        = 'input/train_all/'
    test_data         = 'input/test_jpg/'
    sample_submission = 'input/sample_submission.csv'
    kaggle_csv        = 'input/train.csv'
    external_csv      = 'input/external_data/train.csv'

@path_ingredient.named_config
def old_data():
    train_data = 'input/train_all_old/'

@path_ingredient.capture
def prepair_dir(root, exp_logs, submit):
    create_dir(root + exp_logs)
    create_dir(root + submit)
