from utils import create_dir
from sacred import Ingredient
path_ingredient = Ingredient('path')

@path_ingredient.config
def cfg():
    root              = '/home/dizzyvn/workspace/kaggle/humpback-whale-identification/'
    exp_logs          = 'exp_logs/'
    submit            = 'submit/'
    train_data        = 'input/train/'
    test_data         = 'input/test/'
    sample_submission = 'input/sample_submission.csv'
    train_csv         = 'input/train.csv'
    p2h               = 'input/metadata/p2h.p'
    h2p_prefer        = 'input/metadata/p2h_prefer.p'
    h2ps              = 'input/metadata/h2ps.p'
    p2size            = 'input/metadata/p2size.p'
    h2ws              = 'input/metadata/h2ws.p'
    w2hs              = 'input/metadata/w2hs.p'
    w2ts              = 'input/metadata/w2ts.p'
    t2w               = 'input/metadata/t2w.p'
    train_ps          = 'input/metadata/train_ps.p'
    bbox              = 'input/cropping.csv'

@path_ingredient.capture
def prepair_dir(root, exp_logs, submit):
    create_dir(root + exp_logs)
    create_dir(root + submit)
