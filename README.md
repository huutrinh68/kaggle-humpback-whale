This code use `sacred` as an experiment management tool and a modified version of `minetorch` as learning checkpoints management tool

## Independent

- sacred==0.7.4
- python==3.6.7
- pymongo==3.7.2
- tensorboardX
- omniboard [(Installation guide)](https://vivekratnavel.github.io/omniboard/#/quick-start?id=installation)

## Usage

### TODO

Change root path: replace variable `root` in `path.py` by your project root path

### CONFIG

To print trainning config (default config)

```sh
$python train.py print_config
  INFO - Sample - Running command 'print_config'
  INFO - Sample - Started
  Configuration (modified, added, typechanged, doc):
  debug = False
  max_epochs = 20
  resume = True
  seed = 455565145                   # the random seed for this experiment
  threshold = 0.2
  criterion:
    loss = 'logbce'
    weight = 'log'                   # log / linear / none (default: log)
  data:
    aug_train = ['noop', 'rot90', 'rot180', 'rot270', 'shear', 'flipud', 'fliplr']
    aug_tta = ['noop', 'flipup', 'fliplr', 'rot90', 'rot-90']    # tta aug actions
    augment = True                   # train augment
    batch_size = 40                  # batch size
    image_size = 512                 # image size
    n_channels = 4                   # num of channels
    n_classes = 28                   # num of classes
    n_fold = 5                       # n fold
    n_tta = 3                        # num of tta aug actions
    n_workers = 4                    # num of loader workers
    upsampling = True
  model:
    model = 'resnet18'               # resnet18 / resnet34 / bninception / seres50, default: bninception
  optimizer:
    lr = 0.0001                      # learning rate
    optimizer = 'adam'               # sgd / adam / adamw, default: adam
    weight_decay = 1e-08             # weight_decay
  path:
    exp_logs = 'exp_logs/'
    external_csv = 'input/external_data/train.csv'
    kaggle_csv = 'input/train.csv'
    root = '/home/tran/workspace/new_human_protein_atlas_image_classfication/'
    sample_submission = 'input/sample_submission.csv'
    submit = 'submit/'
    test_data = 'input/test_jpg/'
    train_data = 'input/train_all/'
  INFO - Sample - Completed after 0:00:00
```

### Usage:

Some examples to show how to change experiment setting. You can change all config that listed above.

- Set experiment seed and number of epochs
  - `python main.py with seed=2050 max_epochs=30`
- Run with small data
  - `python main.py with debug=True`
- Resume training
  - `python main.py resume=True [other config]` (config need to be the same to continue previous train)
- Nested config: note that some config is nested in a config scope (data, model, criterion, optimizer, path)
  - `python main.py model.model='resnet34'`
  - `python main.py data.batch_size=40 data.upsampling=False`
  - `python main.py model.model='seres50' optimizer.optimizer=adamw, optimizer.lr=0.001`

### To run omniboard

`omniboard -m localhost:27017:human_protein`

## For more informations

- [Sacred documentation](https://sacred.readthedocs.io/en/latest/quickstart.html)
- [minetorch](https://github.com/louis-she/minetorch)(human-protein)