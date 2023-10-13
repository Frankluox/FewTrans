This repository contains implementation of FewTrans, a few-shot transfer learning benchmark with improved evaluation protocols.


# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```


# Dataset Preparation

## ILSVRC 2012
1. Download `ilsvrc2012_img_train.tar`, from the [ILSVRC2012 website](http://www.image-net.org/challenges/LSVRC/2012/index)
2. Extract it into `ILSVRC2012_img_train/`, which should contain 1000 files, named `n????????.tar`
3. Extract each of `ILSVRC2012_img_train/n????????.tar` in its own directory
    (expected time: \~30 minutes), for instance:

    ```bash
    for FILE in *.tar;
    do
      mkdir ${FILE/.tar/};
      cd ${FILE/.tar/};
      tar xvf ../$FILE;
      cd ..;
    done
    ```

## ILSVRC Sketch
Download [`ImageNet-Sketch.zip`](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA) and extract it into a directory.

## Aircraft
Download [`fgvc-aircraft-2013b.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) and extract it into a directory.


## CUB: 
Download [`CUB_200_2011.tgz`](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and extract it.

## DTD 
Download [`dtd-r1.0.1.tar.gz`](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) and extract it.

## Quick Draw
Download [`quickdraw.zip`](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip) and extract it.

## Fungi 
Download [`fungi_train_val.tgz`](https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz)
    and
    [`train_val_annotations.tgz`](https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz), then extract them into the same directory. It should contain one
    `images/` directory, as well as `train.json` and `val.json`.

## VGG Flower
Download [`102flowers.tgz`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
    and
    [`imagelabels.mat`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat), then extract `102flowers.tgz`, it will create a `jpg/` sub-directory

## CIFAR100:
Download [`cifar-100-python.tar.gz`](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it.

## UCF:
Download [`UCF-101-midframes.zip`](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it.

## EuroSAT:
Download [`EuroSAT.zip`](http://madm.dfki.de/files/sentinel/EuroSAT.zip) and extract it.

## Plant Disease:
Download from [`here`](https://www.kaggle.com/datasets/saroz014/plant-disease).

# Evaluation

## Determining the range of hyperparameters
To evaluate a specific algorithm, first use `write_yaml_search.py` to find coarse-grained hyperparameters on CUB or ImageNet-val, then use `auto_find.py` to find fine-grained hyperparameters. Expand the found hyperparameters to a range, modify `ft_lr_1s`, `ft_lr_2s` and `ft_epochs` in `models/fewshot_finetune_ensemble.py`, and then use `write_yaml_test_with_arg_visual_only.py`(for unimodal algorithms) or `write_yaml_test_with_arg.py`(for multimodal algorithms) to evaluate the methods. Experiments are defined via [yaml](configs) files with the help of [YACS](https://github.com/rbgirshick/yacs) package, following [Swin Transformer](https://github.com/microsoft/Swin-Transformer/blob/main). The basic configurations are defined in `config.py`, overwritten by yaml files.




