# Multi-Label-Classification Without Deep Learning

This repo contains some rough experiments on a relatively small multi-label classification dataset. The goal is to show some of the difficulties that arise
with multi-label classification, and how these difficulties can be overcome using relatively unknown techiques that don't require transfer learning using
large language models. The code is in no way production quality and shouldn't be interpreted as such. For anyone reading, you may find the commentary in notebooks
informative but the code is not.

The main takeaway is that two techniques can be used to drastically improve performance on a multi-label classification dataset:

1. Classifier chains. Classifier chains train One-vs-Rest classifiers on a subset of the labels and then use the classifiers to predict the remaining labels sequentially.
2. Iterative-Stratification. Iterative stratification is a novel sampling technique so that the training set recapitulates the distribution of the labels in multi-label
classification datasets in each fold, thus improving bias-variance tradeoff.

## Structure
Since this repo is mostly a collection of experiments for personal use, it's mostly unstructured. The training pipeline can be found in src/sklearn_trainer.py
and the results

## General setup considerations.

### Installing XGBoost from source

* [Here](https://xgboost.readthedocs.io/en/stable/build.html) is a quick guide to installing XGBoost from source for leveraging GPU.
* Note, if you're using a linux machine for xgboost you may need an earlier gcc compiler. These are helpful links [here](https://askubuntu.com/questions/1039856/downgrade-gnu-compilers-ubuntu-18-04) and [here](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa) In general, you don't want to downgrade for the system,
so follow the links for switching.



