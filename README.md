# Deep Metric Learning

### Learn a deep metric which can be used image retrieval , clustering.
============================

## Pytorch Code for deep metric methods:

- Contrasstive Loss

- Batch-All-Loss and Batch-Hard-Loss

    Loss Functions in [In Defense of Triplet Loss in ReID](https://arxiv.org/abs/1703.07737)

- Lifted Structure Loss
[](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)


- Binomial BinDeviance Loss 

- NCA Loss


   Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure  -Ruslan Salakhutdinov and Geoffrey Hinton

  Though the method was proposed in 2007, It has best performance.

  Recall@1 is 66.3 on  CUB-200-2011 with Dim 512 finetuned on Imagenet-pretrained BN-Inception


## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set

- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online-Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing

- [In-Shop-clothes-Retrieval](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
    For the In-Shop Clothes Retrieval dataset, 3,997 classes with 25,882 images for training.
    And the test set are partitioned to query set with 3,985 classes(14,218 images) and gallery set with 3,985 classes (12,612 images).


## Pretrained models in Pytorch

Pre-trained Inceptionn-BN(inception-v2) used in most deep metric learning papers

Download site: http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth

```bash
wget http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth

mkdir pretrained_models

cp   bn_inception-239d2248.pth    pretrained_models/
```

## Prerequisites

- Computer with Linux or OSX
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.

#### Attention!!
The pre-trained model inception-v2 is transferred from Caffe, it can only  work normally on specific version of Pytorch.
Please create an env as follows:

- Python 
- [PyTorch](http://pytorch.org)  : (0.2.03)
(I have tried 0.3.0 and 0.1.0,  performance is lower than 0.2.03 by 10% on rank@1)

#### Another Attention!!
If you are not required to used inception-BN, you better use my New repository is at https://github.com/bnulihaixia/VGG_dml. 

Performance is nearly the same as BN-inception,  training speed is a bit faster.

which can work normally on pytorch 0.4.0 (the newest stable version)

## Performance of Loss:

I will update this in next month.  The performances of different metric learning losses on the four datasets will be list below. 
Experiment is doing now.  After all the experiments done, I will make a table here. 

### Via some data precessing, Result is much better now.

## Reproducing Car-196 (or CUB-200-2011) experiments

**With  Binomial Deviance Loss  :**

```bash
sh run_train_00.sh
```

## Notice!!!
For the pretrained model of Inception-BN transferred from Caffe can only work normally on torch 0.2.0

# The New repository is at https://github.com/bnulihaixia/VGG_dml
